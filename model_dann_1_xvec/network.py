import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from utils.network_utils import xvecTDNN, X_vector, L2_norm, ReverseLayerF


class time_x_vec_DANN(nn.Module):
    def __init__(self, input_dim, tdnn_embedding_size, triplet_output_size, pair_output_size):
        super(time_x_vec_DANN, self).__init__()

        # (1) Feature Extractor
        self.freq_feature_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=tdnn_embedding_size)

        # (2) Label predictor
        # Layer 1
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('lp_fc1', nn.Linear(tdnn_embedding_size, triplet_output_size))

        # self.label_predictor.add_module('lp_bn1', nn.BatchNorm1d(triplet_output_size))
        # self.label_predictor.add_module('lp_prelu1', nn.PReLU())
        # self.class_classifier.add_module('lp_drop1', nn.Dropout())

        # Layer 2
        # self.class_classifier.add_module('lp_fc2', nn.Linear(128, 64))
        # self.class_classifier.add_module('lp_bn2', nn.BatchNorm1d(64))
        # self.class_classifier.add_module('lp_relu2', nn.ReLU(True))

        # Layer 3
        # self.label_predictor.add_module('lp_fc3', nn.Linear(triplet_output_size, triplet_output_size))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # (3) Domain classifier
        # Layer 1
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_fc1', nn.Linear(pair_output_size, pair_output_size))
        # self.domain_classifier.add_module('dc_bn1', nn.BatchNorm1d(pair_output_size))
        self.domain_classifier.add_module('dc_prelu1', nn.PReLU())

        # Layer 2
        # self.domain_classifier.add_module('dc_fc2', nn.Linear(pair_output_size, pair_output_size))
        # self.domain_classifier.add_module('dc_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        freq_x_vec, freq_tdnn_output, _ = self.freq_feature_extractor(input_data)

        # feature = feature.view(-1, 50 * 4 * 4)
        # reverse_tdnn_output = ReverseLayerF.apply(tdnn_output, alpha)

        class_output = self.label_predictor(freq_x_vec)
        domain_output = self.domain_classifier(freq_x_vec)

        return class_output, domain_output, freq_x_vec


class xvec_dann_triplet(nn.Module):
    def __init__(self, input_dim, embedDim, lpDim, dcDim):
        super(xvec_dann_triplet, self).__init__()
        # (1) Feature Extractor
        self.freq_feature_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=embedDim)
        self.time_feature_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=embedDim)

        # (2) Label predictor
        # Layer 1
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('lp_fc1', nn.Linear(embedDim * 2, lpDim))
        self.label_predictor.add_module('lp_l2norm', L2_norm())
        # self.label_predictor.add_module('lp_bn1', nn.BatchNorm1d(512))

        # self.class_classifier.add_module('lp_drop1', nn.Dropout())

        # ------------------------------ Additional Layer for Test -------------------------------------
        # self.class_classifier.add_module('lp_fc2', nn.Linear(128, 64))
        # self.class_classifier.add_module('lp_bn2', nn.BatchNorm1d(64))
        # self.class_classifier.add_module('lp_relu2', nn.ReLU(True))
        # ----------------------------------------------------------------------------------------------

        # Layer 2
        # self.label_predictor.add_module('lp_fc3', nn.Linear(triplet_output_size, triplet_output_size))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # (3) Domain classifier
        # Layer 1
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_fc1', nn.Linear(embedDim * 2, lpDim))
        self.label_predictor.add_module('dc_l2norm', L2_norm())
        # self.domain_classifier.add_module('dc_bn1', nn.BatchNorm1d(512))
        # self.domain_classifier.add_module('dc_prelu1', nn.PReLU())

        # Layer 2
        # self.domain_classifier.add_module('dc_fc2', nn.Linear(512, pair_output_size))
        # self.domain_classifier.add_module('dc_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        freq_x_vec, freq_tdnn_output, _ = self.freq_feature_extractor(input_data)
        time_x_vec, time_tdnn_output, _ = self.time_feature_extractor(input_data.transpose(1, 2))

        feat_output = torch.cat((freq_x_vec, time_x_vec), dim=1)

        class_output = self.label_predictor(feat_output)
        domain_output = self.domain_classifier(feat_output)

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