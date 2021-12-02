import os
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler

from model_dann_1_xvec.dataset import KnockDataset_train, KnockDataset_val, KnockDataset_pair, load_data, BalancedBatchSampler
from offline_trainer import val_epoch, model_fit
from utils.confusion_matrix_plot import plot_confusion_matrix
import config

# /////////////////////////////////////// Baseline Training & Testing  /////////////////////////////////////////////////
root_dir = '../Knock_dataset/feature_data/stft'
dom = ['exp_data', 'sim_data']
cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

# (1) 数据集提取
# 预读取数据
(src_x_total, src_y_total), (tgt_x_total, tgt_y_total) = load_data(dataset_dir=root_dir, dom=dom, train_flag=1)

# Train
pair_wise_dataset = KnockDataset_pair(
    src_root_data=(src_x_total, src_y_total),
    tgt_root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
batch_sampler = BalancedBatchSampler(
    labels=pair_wise_dataset.exp_label,
    n_classes=pair_wise_dataset.n_classes,
    n_samples=config.NUM_SAMPLES_PER_CLASS)
train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_sampler=batch_sampler)

# Test
src_val_dataset = KnockDataset_val(
    root_data=(src_x_total, src_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))

tgt_val_dataset = KnockDataset_val(
    root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))

val_loader = (src_val_loader, tgt_val_loader)

# (2) 网络、损失函数、优化器
from network import xvec_dann_orig

# 网络模型
freq_size = pair_wise_dataset.data_shape[1]
seq_len = pair_wise_dataset.data_shape[2]
input_dim = (freq_size, seq_len)

model = xvec_dann_orig(
    input_dim=input_dim,
    embedDim=config.EMBEDDING_SIZE,
    n_cls=pair_wise_dataset.n_classes,
    p_dropout=config.P_DROP,
    version=config.XVEC_VERSION
)

if cuda:
    model.cuda()

# 损失函数
loss_fn = torch.nn.NLLLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=config.OFF_INITIAL_LR, weight_decay=config.OFF_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, config.OFF_LR_ADJUST_STEP, gamma=config.OFF_LR_ADJUST_RATIO, last_epoch=-1)

# (3) Baseline model Training & Testing
if config.TRAIN_STAGE:
    model_fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.OFFLINE_EPOCH,
        cuda=cuda)

# ////////////////////////////////////////////// Fine-tuning & Testing /////////////////////////////////////////////////
from model_dann_2_xvec.utils.embedding_visualization import tsne_plot
from model_dann_2_xvec.network import ft_xvec_dann_orig
from model_dann_1_xvec.dataset import KnockDataset_test
from utils.net_train_utils import model_parameter_printing
from online_trainer import netFT_fit


# (1) 创建Fine-tuning数据集
# Query Set
qry_dataset = KnockDataset_test(
    root_data=(src_x_total, src_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
qry_loader = torch.utils.data.DataLoader(qry_dataset, batch_size=len(qry_dataset))

# Support Set
spt_dataset = KnockDataset_test(
    root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
spt_loader = torch.utils.data.DataLoader(spt_dataset, batch_size=config.FINE_TUNE_BATCH)

# (2) 加载Baseline Model
model_path = '../results/output_model'
model_list = list(os.listdir(model_path))
model_list.sort(reverse=True)
model_name = model_list[0]
baseline_model = torch.load(os.path.join(model_path, model_name))


# (3) 验证基线模型 （混淆矩阵 + t-SNE可视化）
src_train_dataset = KnockDataset_train(
    root_data=(src_x_total, src_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
tgt_train_dataset = KnockDataset_train(
    root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)

val_loader_list = [
    torch.utils.data.DataLoader(src_train_dataset, batch_size=len(src_train_dataset)),
    torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset)),
    torch.utils.data.DataLoader(tgt_train_dataset, batch_size=len(tgt_train_dataset)),
    torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))]

accu, cm = val_epoch(val_loader_list[3], baseline_model, cuda)

plot_confusion_matrix(cm=cm,
                      save_path=os.path.join('../results', 'output_confusion_matrix', model_name + '_experiment' + '.png'),
                      classes=[str(i) for i in pair_wise_dataset.pair_label_set])

print('\nBaseline model validation accuracy: %0.2f %%' % (accu * 100))

# tsne_plot(
#     src_dataset=(src_val_dataset.val_data, src_val_dataset.val_label),
#     tgt_dataset=(tgt_val_dataset.val_data, tgt_val_dataset.val_label),
#     model=baseline_model,
#     model_name=model_name,
#     cuda=cuda)

# (4) 定义Fine-tuning网络及可训练参数
fine_tuned_model = ft_xvec_dann_orig(
    baseModel=baseline_model,
    n_class=len(config.SUPPORT_SET_LABEL),
    version=config.XVEC_VERSION)

fixed_module = ['feature_extractor', 'domain_classifier']
# fixed_module = ['domain_classifier']
for name, param in fine_tuned_model.named_parameters():
    net_module = name.split('.')[0]
    if net_module in fixed_module:
        param.requires_grad = False

model_parameter_printing(fine_tuned_model)  # 打印网络参数

# (5) Re-initialize Loss_func and Optimizer
loss_fn = torch.nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.ON_INITIAL_LR, weight_decay=config.ON_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, config.ON_LR_ADJUST_STEP, gamma=config.ON_LR_ADJUST_RATIO, last_epoch=-1)

# (6) Fine-tuning & Testing
if config.FINE_TUNE_STAGE:
    netFT_fit(
        train_loader=spt_loader,
        test_loader=qry_loader,
        support_label_set=config.SUPPORT_SET_LABEL,
        model=fine_tuned_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.ONLINE_EPOCH,
        cuda=cuda
    )