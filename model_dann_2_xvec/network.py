import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from model_dann_1_xvec.network import L2_norm, X_vector, ReverseLayerF


class gan_xvec(nn.Module):
    def __init__(self, input_dim, xvec_embed_len, n_cls=0):
        super(gan_xvec, self).__init__()

        # (1) Feature Extractor
        self.freq_feat_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=xvec_embed_len)
        self.time_feat_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=xvec_embed_len)

        self.l2_norm = L2_norm()

        # (2) Domain Classifier
        self.dmn_clr = nn.Sequential()
        # Layer 1
        self.dmn_clr.add_module('dc_fc1', nn.Linear(xvec_embed_len * 1, xvec_embed_len * 4))
        self.dmn_clr.add_module('dc_bn1', nn.BatchNorm1d(xvec_embed_len * 4))
        self.dmn_clr.add_module('dc_prelu1', nn.PReLU())
        # Layer 2
        self.dmn_clr.add_module('dc_fc2', nn.Linear(xvec_embed_len * 4, xvec_embed_len))
        self.dmn_clr.add_module('dc_bn2', nn.BatchNorm1d(xvec_embed_len))
        self.dmn_clr.add_module('dc_prelu2', nn.PReLU())
        # Layer 3
        self.dmn_clr.add_module('dc_fc3', nn.Linear(xvec_embed_len, 2))
        self.dmn_clr.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

        # (3) Label Predictor
        self.lbl_pred = nn.Sequential()
        # Layer 1
        self.lbl_pred.add_module('lp_fc1', nn.Linear(xvec_embed_len * 1, xvec_embed_len * 4))
        self.lbl_pred.add_module('lp_bn1', nn.BatchNorm1d(xvec_embed_len * 4))
        self.lbl_pred.add_module('lp_prelu1', nn.PReLU())
        # Layer 2
        self.lbl_pred.add_module('lp_fc2', nn.Linear(xvec_embed_len * 4, xvec_embed_len))
        self.lbl_pred.add_module('lp_bn2', nn.BatchNorm1d(xvec_embed_len))
        self.lbl_pred.add_module('lp_prelu2', nn.PReLU())
        # Layer 3
        self.lbl_pred.add_module('dc_fc3', nn.Linear(xvec_embed_len, n_cls))
        self.lbl_pred.add_module('dc_logsoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        _, freq_feat, _ = self.freq_feat_extractor(input_data)
        # _, time_feat, _ = self.time_feat_extractor(input_data.transpose(1, 2))

        # feat_output = torch.cat((freq_feat, time_feat), dim=1)
        feat_output = self.l2_norm(freq_feat)

        lp_output = self.lbl_pred(feat_output)

        rev_feat = ReverseLayerF.apply(feat_output, alpha)
        dc_output = self.dmn_clr(rev_feat)

        return feat_output, lp_output, dc_output






