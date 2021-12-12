import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from utils.network_utils import xvecTDNN, X_vector, L2_norm, ReverseLayerF


class xvec_dann_orig(nn.Module):
    def __init__(self, input_dim, args, freq_time=False, n_cls=0, version=1):
        super(xvec_dann_orig, self).__init__()
        self.freq_time = freq_time
        self.version = version

        embedDim = args['EMBED_SIZE']
        fc1_output_dim = args['FC1_OUT_DIM']
        fc2_output_dim = args['FC2_OUT_DIM']

        # (1) Feature Extractor
        if version == 1:
            if self.freq_time:
                self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=embedDim)
                self.time_feat_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=embedDim)
            else:
                self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=embedDim)
        elif version == 2:
            self.freq_feat_extractor = xvecTDNN(inputDim=input_dim[0], args=args)

        self.l2_norm = L2_norm()

        # (2) Domain Classifier
        self.dmn_clr = nn.Sequential()
        # Layer 1
        self.dmn_clr.add_module('dc_fc1', nn.Linear(embedDim * 1, embedDim * fc1_output_dim))
        self.dmn_clr.add_module('dc_bn1', nn.BatchNorm1d(embedDim * fc1_output_dim))
        self.dmn_clr.add_module('dc_prelu1', nn.PReLU())
        # Layer 2
        self.dmn_clr.add_module('dc_fc2', nn.Linear(embedDim * fc1_output_dim, embedDim * fc2_output_dim))
        self.dmn_clr.add_module('dc_bn2', nn.BatchNorm1d(embedDim * fc2_output_dim))
        self.dmn_clr.add_module('dc_prelu2', nn.PReLU())
        # Layer 3
        self.dmn_clr.add_module('dc_fc3', nn.Linear(embedDim * fc2_output_dim, 2))
        self.dmn_clr.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

        # (3) Label Predictor
        self.lb_pred = nn.Sequential()
        # Layer 1
        self.lb_pred.add_module('lp_fc1', nn.Linear(embedDim * 1, embedDim * fc1_output_dim))
        self.lb_pred.add_module('lp_bn1', nn.BatchNorm1d(embedDim * fc1_output_dim))
        self.lb_pred.add_module('lp_prelu1', nn.PReLU())
        # Layer 2
        self.lb_pred.add_module('lp_fc2', nn.Linear(embedDim * fc1_output_dim, embedDim * fc2_output_dim))
        self.lb_pred.add_module('lp_bn2', nn.BatchNorm1d(embedDim * fc2_output_dim))
        self.lb_pred.add_module('lp_prelu2', nn.PReLU())
        # Layer 3
        self.lb_pred.add_module('lp_fc3', nn.Linear(embedDim * fc2_output_dim, n_cls))
        self.lb_pred.add_module('lp_logsoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha, eps):
        if self.version == 1:
            # Note: input size(batch, seq_len, input_features)
            input_data = input_data.permute(0, 2, 1)
            if self.freq_time:
                _, freq_feat, _ = self.freq_feat_extractor(input_data)
                _, time_feat, _ = self.time_feat_extractor(input_data.transpose(1, 2))
                feat_output = torch.cat((freq_feat, time_feat), dim=1)
            else:
                feat_output, _, _ = self.freq_feat_extractor(input_data)
        elif self.version == 2:
            # Note: x must be (batch_size, feat_dim, chunk_len)
            feat_output = self.freq_feat_extractor(input_data, eps)

        # L2正则化
        feat_output = self.l2_norm(feat_output)

        # Label Predictor 前向传播
        lp_output = self.lb_pred(feat_output)

        # Domain Regressor 前向传播
        rev_feat = ReverseLayerF.apply(feat_output, alpha)
        dc_output = self.dmn_clr(rev_feat)

        return feat_output, lp_output, dc_output


class ft_xvec_dann_orig(nn.Module):
    def __init__(self, baseModel, args, n_class, version=1):
        super(ft_xvec_dann_orig, self).__init__()
        self.version = version

        embedDim = args['EMBED_SIZE']
        fc1_output_dim = args['FC1_OUT_DIM']
        fc2_output_dim = args['FC2_OUT_DIM']

        # (1) 提取预训练模型
        # Feature Extractor
        if version == 1:
            self.feature_extractor = list(baseModel.children())[0]
        elif version == 2:
            self.feature_extractor = list(baseModel.children())[0]

        # Label Predictor
        # self.lp_new = list(baseModel.children())[3]
        # self.lp_new.dc_fc3 = nn.Linear(embedDim * fc2_output_dim, n_class)

        # (2) 新增Label predictor
        self.lp_new = nn.Sequential()
        # Layer 1
        self.lp_new.add_module('n_lp_fc1', nn.Linear(embedDim, embedDim * fc1_output_dim))
        self.lp_new.add_module('n_lp_bn1', nn.BatchNorm1d(embedDim * fc1_output_dim))
        self.lp_new.add_module('n_lp_prelu1', nn.PReLU())
        # Layer 2
        self.lp_new.add_module('n_lp_fc2', nn.Linear(embedDim * fc1_output_dim, embedDim * fc2_output_dim))
        self.lp_new.add_module('n_lp_bn2', nn.BatchNorm1d(embedDim * fc2_output_dim))
        self.lp_new.add_module('n_lp_prelu2', nn.PReLU())
        # Layer 3
        self.lp_new.add_module('n_lp_fc3', nn.Linear(embedDim * fc2_output_dim, n_class))
        self.lp_new.add_module('n_lp_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        '''
        (1) 方案1不进行Domain classifier的训练，所以前向传播不需要Domain classifier的计算
        (2) __init__中仍然保存了Domain classifier和Feature extractor的参数，这是为了以后再次更新模型
        '''
        if self.version == 1:
            _, feat_output, _ = self.feature_extractor(input_data)
            class_output = self.lp_new(feat_output)
        elif self.version == 2:
            feat_output = self.feature_extractor(input_data, 0)
            class_output = self.lp_new(feat_output)

        return class_output
