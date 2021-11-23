from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
from model_dann_1_xvec.dataset import *
from trainer import *
from model_dann_1_xvec.online_trainer import *
from model_dann_1_xvec.config import *
from model_dann_1_xvec.network import X_vector


root_dir = '../Knock_dataset/feature_data/fbank_denoised_data'
source_domain = 'exp_data'
target_domain = 'sim_data'
cuda = torch.cuda.is_available()

# (1) 数据集
src_train_dataset = KnockDataset_train(root_dir, source_domain, SUPPORT_SET_LABEL)
tgt_train_dataset = KnockDataset_train(root_dir, target_domain, SUPPORT_SET_LABEL)
src_val_dataset = KnockDataset_val(root_dir, source_domain, SUPPORT_SET_LABEL)
tgt_val_dataset = KnockDataset_val(root_dir, target_domain, SUPPORT_SET_LABEL)

src_train_n_classes = src_train_dataset.n_classes
tgt_train_n_classes = tgt_train_dataset.n_classes

x_vec_train_batch_size = 100
src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=int(x_vec_train_batch_size/2), shuffle=True)
tgt_train_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=int(x_vec_train_batch_size/2), shuffle=True)
src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))
tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))

offline_train_loader = (src_train_loader, tgt_train_loader)
offline_val_loader = (src_val_loader, tgt_val_loader)

# (2) 网络、损失函数、优化器

freq_size = src_train_dataset.x_data_total.shape[2]
seq_len = src_train_dataset.x_data_total.shape[1]
input_dim = (freq_size, seq_len)

# 网络模型
model = X_vector(input_dim=input_dim[0], tdnn_embedding_size=EMBEDDING_SIZE, n_class=src_train_n_classes)
if cuda:
    model.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=OFF_INITIAL_LR, weight_decay=OFF_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, OFF_LR_ADJUST_STEP, gamma=OFF_LR_ADJUST_RATIO, last_epoch=-1)

non_transfer_base_line_fit(
    train_loader=offline_train_loader,
    val_loader=offline_val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=OFFLINE_EPOCH,
    cuda=cuda
)