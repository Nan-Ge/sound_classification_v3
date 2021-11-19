import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from model_dann_1_xvec.network import *


class gan_xvec(nn.Module):
    def __init__(self, input_dim, xvec_embed_len):
        super(gan_xvec, self).__init__()

        # (1) Feature Extractor
        self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=xvec_embed_len)
        self.time_feat_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=xvec_embed_len)

        # (2) Domain Classifier
        self.dmn_clr = nn.Sequential()
        # Layer 1
        self.dmn_clr.add_module('dc_fc1', nn.Linear(xvec_embed_len * 2, xvec_embed_len * 2))
        self.dmn_clr.add_module('dc_bn1', nn.BatchNorm1d(xvec_embed_len * 2))
        self.dmn_clr.add_module('dc_prelu1', nn.PReLU())
        self.dmn_clr.add_module('dc_drop1', nn.Dropout())
        # Layer 2
        self.dmn_clr.add_module('dc_fc2', nn.Linear(xvec_embed_len * 2, xvec_embed_len))
        self.dmn_clr.add_module('dc_bn2', nn.BatchNorm1d(xvec_embed_len))
        self.dmn_clr.add_module('dc_prelu2', nn.PReLU())
        self.dmn_clr.add_module('dc_drop2', nn.Dropout())
        # Layer 3
        self.dmn_clr.add_module('dc_fc3', nn.Linear(xvec_embed_len, 2))
        self.dmn_clr.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        _, src_freq_feat, _ = self.freq_feat_extractor(input_data[0])
        _, src_time_feat, _, _ = self.time_feat_extractor(input_data[0].transpose(1, 2))
        src_feat_output = torch.cat((src_freq_feat, src_time_feat), dim=1)

        _, tgt_freq_feat, _ = self.freq_feat_extractor(input_data[0])
        _, tgt_time_feat, _, _ = self.time_feat_extractor(input_data[0].transpose(1, 2))
        tgt_feat_output = torch.cat((tgt_freq_feat, tgt_time_feat), dim=1)

        diff_feat = src_feat_output - tgt_feat_output  # src与tgt特征做差
        rev_diff_feat = ReverseLayerF.apply(diff_feat, alpha)

        dc_output = self.dmn_clr(rev_diff_feat)

        return src_feat_output, tgt_feat_output, dc_output






