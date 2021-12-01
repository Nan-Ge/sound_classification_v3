import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function


# ---------------------------------- Custom Components ------------------------------------
class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ---------------------------------- Version 1 --------------------------------------------
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
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=256, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=256, output_dim=256, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=256, output_dim=256, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=256, output_dim=256, context_size=1, dilation=3, dropout_p=0.3)
        self.segment6 = nn.Linear(256 * 2, tdnn_embedding_size)
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

        # Fully-connected
        x_vec = self.segment6(stat_pooling)
        segment7_out = self.segment7(x_vec)
        linear_output = self.output(segment7_out)

        # Softmax
        # predictions = self.softmax(linear_output)

        return x_vec, segment7_out, linear_output


# ---------------------------------- Version 2 --------------------------------------------
class xvecTDNN(nn.Module):

    def __init__(self, inputDim, embedDim, p_dropout):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=inputDim, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, embedDim)
        self.bn_fc1 = nn.BatchNorm1d(embedDim, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        # self.fc2 = nn.Linear(512, 512)
        # self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        # self.dropout_fc2 = nn.Dropout(p=p_dropout)

        # self.fc3 = nn.Linear(512, numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))

        # x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        # x = self.fc3(x)

        return x
