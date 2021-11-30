import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from utils.network_utils import xvecTDNN, X_vector, L2_norm, ReverseLayerF


class gan_xvec(nn.Module):
    def __init__(self, input_dim, embedDim, p_dropout, freq_time=False, n_cls=0):
        super(gan_xvec, self).__init__()
        self.freq_time = freq_time
        # (1) Feature Extractor

        # ----- Version 1 -----
        if self.freq_time:
            self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=embedDim)
            self.time_feat_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=embedDim)
        else:
            self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=embedDim)

        # ----- Version 2 -----
        # self.freq_feat_extractor = xvecTDNN(inputDim=input_dim[0], embedDim=embedDim, p_dropout=p_dropout)

        self.l2_norm = L2_norm()

        # (2) Domain Classifier
        self.dmn_clr = nn.Sequential()
        # Layer 1
        self.dmn_clr.add_module('dc_fc1', nn.Linear(embedDim * 1, embedDim * 2))
        self.dmn_clr.add_module('dc_bn1', nn.BatchNorm1d(embedDim * 2))
        self.dmn_clr.add_module('dc_prelu1', nn.PReLU())
        # Layer 2
        self.dmn_clr.add_module('dc_fc2', nn.Linear(embedDim * 2, embedDim))
        self.dmn_clr.add_module('dc_bn2', nn.BatchNorm1d(embedDim))
        self.dmn_clr.add_module('dc_prelu2', nn.PReLU())
        # Layer 3
        self.dmn_clr.add_module('dc_fc3', nn.Linear(embedDim, 2))
        self.dmn_clr.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

        # (3) Label Predictor
        self.lbl_pred = nn.Sequential()
        # Layer 1
        self.lbl_pred.add_module('lp_fc1', nn.Linear(embedDim * 1, embedDim * 2))
        self.lbl_pred.add_module('lp_bn1', nn.BatchNorm1d(embedDim * 2))
        self.lbl_pred.add_module('lp_prelu1', nn.PReLU())
        # Layer 2
        self.lbl_pred.add_module('lp_fc2', nn.Linear(embedDim * 2, embedDim))
        self.lbl_pred.add_module('lp_bn2', nn.BatchNorm1d(embedDim))
        self.lbl_pred.add_module('lp_prelu2', nn.PReLU())
        # Layer 3
        self.lbl_pred.add_module('dc_fc3', nn.Linear(embedDim, n_cls))
        self.lbl_pred.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha, eps):
        # ----- Version 1 -----
        input_data = input_data.permute(0, 2, 1)
        if self.freq_time:
            _, freq_feat, _ = self.freq_feat_extractor(input_data)
            _, time_feat, _ = self.time_feat_extractor(input_data.transpose(1, 2))
            feat_output = torch.cat((freq_feat, time_feat), dim=1)
        else:
            _, feat_output, _ = self.freq_feat_extractor(input_data)

        # ----- Version 2 -----
        # feat_output = self.freq_feat_extractor(input_data, eps)

        # L2正则化
        feat_output = self.l2_norm(feat_output)

        # Label Predictor 前向传播
        lp_output = self.lbl_pred(feat_output)

        # Domain Regressor 前向传播
        rev_feat = ReverseLayerF.apply(feat_output, alpha)
        dc_output = self.dmn_clr(rev_feat)

        return feat_output, lp_output, dc_output
