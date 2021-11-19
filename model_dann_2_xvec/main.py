import torch.optim as optim
import torch.utils.data
import random
from torch.optim import lr_scheduler

from model_dann_1_xvec.dataset import *
import model_dann_1_xvec.config as config
from trainer import *
from utils.confusion_matrix_plot import plot_confusion_matrix

# /////////////////////////////////////// Baseline Training & Testing  /////////////////////////////////////////////////
root_dir = '../Knock_dataset/feature_data/fbank_denoised_data'
src_dmn = 'exp_data'
tgt_dmn = 'sim_data'
cuda = torch.cuda.is_available()

# (1) 数据集提取
# Train
pair_wise_dataset = KnockDataset_pair(root_dir, support_label_set=config.SUPPORT_SET_LABEL)
batch_sampler = BalancedBatchSampler(labels=pair_wise_dataset.exp_label,
                                     n_classes=pair_wise_dataset.n_classes,
                                     n_samples=config.NUM_SAMPLES_PER_CLASS)
train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_sampler=batch_sampler)
# Test
src_val_dataset = KnockDataset_val(root_dir, src_dmn, SUPPORT_SET_LABEL)
src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))

tgt_val_dataset = KnockDataset_val(root_dir, tgt_dmn, SUPPORT_SET_LABEL)
tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))

val_loader = (src_val_loader, tgt_val_loader)

# (2) 网络、损失函数、优化器
from network import gan_xvec
from model_dann_1_xvec.losses import OnlineTripletLoss
from model_dann_1_xvec.loss_utils import SemihardNegativeTripletSelector

# 网络模型
freq_size = pair_wise_dataset.exp_data.shape[2]
seq_len = pair_wise_dataset.exp_data.shape[1]
input_dim = (freq_size, seq_len)

model = gan_xvec(input_dim=input_dim, xvec_embed_len=EMBEDDING_SIZE)
if cuda:
    model.cuda()

# 损失函数
trip_loss = OnlineTripletLoss(config.TRIPLET_MARGIN, SemihardNegativeTripletSelector(config.TRIPLET_MARGIN))
dc_loss = torch.nn.NLLLoss()
loss_fn = (trip_loss, dc_loss)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=OFF_INITIAL_LR, weight_decay=OFF_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, OFF_LR_ADJUST_STEP, gamma=OFF_LR_ADJUST_RATIO, last_epoch=-1)

# (3) Baseline model Training & Testing
model_fit(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=OFFLINE_EPOCH,
    cuda=cuda)


