import torch.nn as nn
from Knock_DANN.functions import ReverseLayerF
import torch.nn.functional as F
import torch


class TDNN(nn.Module):
    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, batch_norm=True,
                 dropout_p=0.2):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.context_size, self.input_dim), stride=(1, self.input_dim), dilation=(self.dilation, 1))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        return x


class X_vector(nn.Module):
    def __init__(self, input_dim, tdnn_embedding_size, n_class=2):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3, dropout_p=0.5)
        self.segment6 = nn.Linear(1024, tdnn_embedding_size)
        self.segment7 = nn.Linear(tdnn_embedding_size, tdnn_embedding_size)

        self.output = nn.Linear(tdnn_embedding_size, n_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        # TDNN feature extraction
        tdnn1_out = self.tdnn1(input_data)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)

        # Stat Pool
        mean = torch.mean(tdnn5_out, 1)
        std = torch.var(tdnn5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)

        # sound embedding
        x_vec = self.segment6(stat_pooling)
        segment7_out = self.segment7(x_vec)
        linear_output = self.output(segment7_out)
        predictions = self.softmax(linear_output)
        return x_vec, x_vec, linear_output


class time_x_vec_DANN(nn.Module):
    def __init__(self, input_dim, tdnn_embedding_size, triplet_output_size, pair_output_size):
        super(time_x_vec_DANN, self).__init__()

        # (1) Feature Extractor
        self.freq_feature_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=tdnn_embedding_size)

        # (2) Label predictor
        # Layer 1
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('lp_fc1', nn.Linear(tdnn_embedding_size, 256))
        self.label_predictor.add_module('lp_bn1', nn.BatchNorm1d(256))
        self.label_predictor.add_module('lp_prelu1', nn.PReLU())
        # self.class_classifier.add_module('lp_drop1', nn.Dropout())

        # Layer 2
        # self.class_classifier.add_module('lp_fc2', nn.Linear(128, 64))
        # self.class_classifier.add_module('lp_bn2', nn.BatchNorm1d(64))
        # self.class_classifier.add_module('lp_relu2', nn.ReLU(True))

        # Output Layer
        self.label_predictor.add_module('lp_fc3', nn.Linear(256, triplet_output_size))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # (3) Domain classifier
        # Layer 1
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_fc1', nn.Linear(tdnn_embedding_size, 256))
        self.domain_classifier.add_module('dc_bn1', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('dc_prelu1', nn.PReLU())

        # Layer 2
        self.domain_classifier.add_module('dc_fc2', nn.Linear(256, pair_output_size))
        # self.domain_classifier.add_module('dc_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        freq_x_vec, freq_tdnn_output, _ = self.freq_feature_extractor(input_data)

        # feature = feature.view(-1, 50 * 4 * 4)
        # reverse_tdnn_output = ReverseLayerF.apply(tdnn_output, alpha)

        class_output = self.label_predictor(freq_x_vec)
        domain_output = self.domain_classifier(freq_x_vec)

        return class_output, domain_output, freq_x_vec


class time_freq_x_vec_DANN(nn.Module):
    def __init__(self, input_dim, tdnn_embedding_size, triplet_output_size, pair_output_size):
        super(time_freq_x_vec_DANN, self).__init__()
        # (1) Feature Extractor
        self.freq_feature_extractor = X_vector(input_dim=input_dim[0], tdnn_embedding_size=tdnn_embedding_size)
        self.time_feature_extractor = X_vector(input_dim=input_dim[1], tdnn_embedding_size=tdnn_embedding_size)

        # (2) Label predictor
        # Layer 1
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('lp_fc1', nn.Linear(tdnn_embedding_size * 2, 512))
        self.label_predictor.add_module('lp_bn1', nn.BatchNorm1d(512))
        self.label_predictor.add_module('lp_prelu1', nn.PReLU())
        # self.class_classifier.add_module('lp_drop1', nn.Dropout())

        # ------------------------------ Additional Layer for Test -------------------------------------
        # self.class_classifier.add_module('lp_fc2', nn.Linear(128, 64))
        # self.class_classifier.add_module('lp_bn2', nn.BatchNorm1d(64))
        # self.class_classifier.add_module('lp_relu2', nn.ReLU(True))
        # ----------------------------------------------------------------------------------------------

        # Layer 2
        self.label_predictor.add_module('lp_fc3', nn.Linear(512, triplet_output_size))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # (3) Domain classifier
        # Layer 1
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_fc1', nn.Linear(tdnn_embedding_size * 2, 512))
        self.domain_classifier.add_module('dc_bn1', nn.BatchNorm1d(512))
        self.domain_classifier.add_module('dc_prelu1', nn.PReLU())

        # Layer 2
        self.domain_classifier.add_module('dc_fc2', nn.Linear(512, pair_output_size))
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