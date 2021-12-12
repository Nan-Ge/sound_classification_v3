import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from utils.network_utils import xvecTDNN, X_vector, L2_norm, ReverseLayerF


class xvec_dann_triplet(nn.Module):
    def __init__(self, input_dim, args):
        super(xvec_dann_triplet, self).__init__()
        self.freq_time = args['FREQ_TIME']

        embedDim = args['EMBED_SIZE'] * self.freq_time

        fc1_output_dim = args['FC1_OUT_DIM']
        fc2_output_dim = args['FC2_OUT_DIM']

        lp_layers = args['LP_LAYERS']
        dc_layers = args['DC_LAYERS']

        lp_dim = args['LP_DIM']
        dc_dim = args['DC_DIM']

        # (1) Feature Extractor
        if self.freq_time == 1:
            self.freq_feat_extractor = xvecTDNN(inputDim=input_dim[0], args=args)
        elif self.freq_time == 2:
            self.freq_feat_extractor = xvecTDNN(inputDim=input_dim[0], args=args)
            self.time_feat_extractor = xvecTDNN(inputDim=input_dim[1], args=args)

        # (2) Label predictor
        if lp_layers == 1:
            # One layer lpl_pred
            self.lp = nn.Sequential()

            self.lp.add_module('lp_fc1', nn.Linear(embedDim, lp_dim))
            # self.lp.add_module('lp_l2norm', L2_norm())
        elif lp_layers == 3:
            # Three layers lpl_pred
            self.lp = nn.Sequential()

            self.lp.add_module('lp_fc1', nn.Linear(embedDim, embedDim * fc1_output_dim))
            self.lp.add_module('lp_bn1', nn.BatchNorm1d(embedDim * fc1_output_dim))
            self.lp.add_module('lp_prelu1', nn.PReLU())

            self.lp.add_module('lp_fc2', nn.Linear(embedDim * fc1_output_dim, embedDim * fc2_output_dim))
            self.lp.add_module('lp_bn2', nn.BatchNorm1d(embedDim * fc2_output_dim))
            self.lp.add_module('lp_prelu2', nn.PReLU())

            self.lp.add_module('lp_fc3', nn.Linear(embedDim * fc2_output_dim, lp_dim))
            # self.lp.add_module('lp_l2norm', L2_norm())

        # (3) Domain classifier
        if dc_layers == 1:
            # One layer dc_clr
            self.dc = nn.Sequential()

            self.dc.add_module('dc_fc1', nn.Linear(embedDim, lp_dim))
            # self.dc.add_module('dc_l2norm', L2_norm())
        elif dc_layers == 3:
            # Three layers dc_clr
            self.dc = nn.Sequential()

            self.dc.add_module('lp_fc1', nn.Linear(embedDim, embedDim * fc1_output_dim))
            self.dc.add_module('lp_bn1', nn.BatchNorm1d(embedDim * fc1_output_dim))
            self.dc.add_module('lp_prelu1', nn.PReLU())

            self.dc.add_module('lp_fc2', nn.Linear(embedDim * fc1_output_dim, embedDim * fc2_output_dim))
            self.dc.add_module('lp_bn2', nn.BatchNorm1d(embedDim * fc2_output_dim))
            self.dc.add_module('lp_prelu2', nn.PReLU())

            self.dc.add_module('lp_fc3', nn.Linear(embedDim * fc2_output_dim, lp_dim))
            # self.dc.add_module('lp_l2norm', L2_norm())

    def forward(self, input_data, eps):
        if self.freq_time == 1:
            feat_output = self.freq_feat_extractor(input_data, eps)
        elif self.freq_time == 2:
            freq_feat = self.freq_feat_extractor(input_data, eps)
            time_feat = self.time_feat_extractor(input_data.transpose(1, 2), eps)
            feat_output = torch.cat((freq_feat, time_feat), dim=1)

        class_output = self.lp(feat_output)
        domain_output = self.dc(feat_output)

        return class_output, domain_output, feat_output


class fine_tuned_DANN_triplet_Net(nn.Module):
    def __init__(self, model, support_set_n_class):
        super(fine_tuned_DANN_triplet_Net, self).__init__()
        # (1) 提取预训练的Feature extractor
        self.feature_extractor = list(model.children())[0]
        # self.domain_classifier = list(model.children())[2]

        self.tdnn_embedding_size = self.feature_extractor.segment7.out_features

        # (2) 新增Label predictor
        self.label_predictor_new = nn.Sequential()
        # Layer 1
        self.label_predictor_new.add_module('n_lp_fc1', nn.Linear(self.tdnn_embedding_size, 128))
        self.label_predictor_new.add_module('n_lp_bn1', nn.BatchNorm1d(128))
        self.label_predictor_new.add_module('n_lp_prelu1', nn.PReLU())

        # Layer 2
        self.label_predictor_new.add_module('n_lp_fc2', nn.Linear(128, 128))
        self.label_predictor_new.add_module('n_lp_bn2', nn.BatchNorm1d(128))
        self.label_predictor_new.add_module('n_lp_prelu2', nn.PReLU())

        # Layer 3
        self.label_predictor_new.add_module('n_lp_fc3', nn.Linear(128, support_set_n_class))
        # self.label_predictor_new.add_module('n_lp_fc3', nn.Linear(self.tdnn_embedding_size, support_set_n_class))
        self.label_predictor_new.add_module('n_lp_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        '''
        (1) 方案1不进行Domain classifier的训练，所以前向传播不需要Domain classifier的计算
        (2) __init__中仍然保存了Domain classifier和Feature extractor的参数，这是为了以后再次更新模型
        '''
        x_vec, tdnn_output, _ = self.feature_extractor(input_data)
        class_output = self.label_predictor_new(tdnn_output)

        return class_output, x_vec