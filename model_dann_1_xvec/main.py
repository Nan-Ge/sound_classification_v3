from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
import random
from model_dann_1_xvec.dataset import *
from offline_trainer import *
from online_trainer import *
from model_dann_1_xvec.config import *
from utils.confusion_matrix_plot import plot_confusion_matrix


# /////////////////////////////////////// Baseline Training & Testing  /////////////////////////////////////////////////
root_dir = './Knock_dataset/feature_data/fbank_denoised_data'
src_dmn = 'exp_data'
tgt_dmn = 'sim_data'

cuda = torch.cuda.is_available()
kwargs = {}

# (1) 数据集
'''
5个数据集：
1、source_train_dataset：
    除去support_label_set中所有标签后的 / 源域的 / 所有样本；
    用于离线阶段 / label predictor的Triplet-loss；
2、target_train_dataset：
    除去support_label_set中所有标签后的 / 目标域的 / 所有样本；
    用于离线阶段 / label predictor的Triplet-loss；
3、pair_wise_dataset：
    除去support_label_set中所有标签后的 / 源域和目标域的 / 成对的 / 所有样本；
    用于在线阶段 / domain classifier的pair-wise loss；
4、support_dataset：
    support_label_set中所有标签的 / 目标域的 / 所有样本；
    用于在线阶段 / 模型的微调；
5、test_dataset：
    support_label_set中所有标签的 / 源域的 / 所有样本；
    用于在线阶段 / Label predictor的准确性测试；
'''

src_train_dataset = KnockDataset_train(root_dir, src_dmn, SUPPORT_SET_LABEL)
tgt_train_dataset = KnockDataset_train(root_dir, tgt_dmn, SUPPORT_SET_LABEL)

src_val_dataset = KnockDataset_val(root_dir, src_dmn, SUPPORT_SET_LABEL)
tgt_val_dataset = KnockDataset_val(root_dir, tgt_dmn, SUPPORT_SET_LABEL)

pair_wise_dataset = KnockDataset_pair(root_dir, support_label_set=SUPPORT_SET_LABEL)

support_dataset = KnockDataset_test(root_dir, tgt_dmn, SUPPORT_SET_LABEL)
query_dataset = KnockDataset_test(root_dir, src_dmn, SUPPORT_SET_LABEL)

print("数据集划分情况：")
print("Src_Triplet_Train(%d), Src_Val(%d), Tgt_Triplet_Train(%d), Tgt_Val(%d)" % (len(src_train_dataset.train_label),
                                                                                  len(src_val_dataset.val_label),
                                                                                  len(tgt_train_dataset.train_label),
                                                                                  len(tgt_val_dataset.val_label)))

# DataLoader
'''
Online pair selection: We'll create mini batches by sampling labels that will be present in the mini batch and number
of examples from each class, 生成mini-batch的大小为 n_classes * n_samples_per_class
'''
src_train_n_classes = src_train_dataset.n_classes
tgt_train_n_classes = tgt_train_dataset.n_classes
test_n_classes = query_dataset.n_classes

# --------- DataLoader for Offline-Stage --------------
# Source Train
src_train_batch_sampler = BalancedBatchSampler(src_train_dataset.train_label, n_classes=src_train_n_classes, n_samples=NUM_SAMPLES_PER_CLASS)
src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_sampler=src_train_batch_sampler, **kwargs)
# Target Train
tgt_train_batch_sampler = BalancedBatchSampler(tgt_train_dataset.train_label, n_classes=tgt_train_n_classes, n_samples=NUM_SAMPLES_PER_CLASS)
tgt_train_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_sampler=tgt_train_batch_sampler, **kwargs)
# Source Validation
src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))
# Target Validation
tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))
# Pair-wise Train
pair_wise_train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_size=PAIR_WISE_BATCH)

# --------- DataLoader for Online-Stage --------------
online_query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=len(query_dataset))  # Online query
fine_tuning_set_loader = torch.utils.data.DataLoader(support_dataset, batch_size=FINE_TUNE_BATCH, shuffle=True)  # Online support

# --------------- DataLoader Summary -----------------
offline_train_loader = (src_train_loader, tgt_train_loader, pair_wise_train_loader)
offline_val_loader = (src_val_loader, tgt_val_loader)


# (2) 网络、损失函数、优化器
from model_dann_1_xvec.network import time_freq_x_vec_DANN
from model_dann_1_xvec.losses import OnlineTripletLoss, PairWiseLoss
from model_dann_1_xvec.loss_utils import SemihardNegativeTripletSelector, KnockPointPairSelector  # Strategies for selecting triplets within a minibatch

# 网络模型
freq_size = src_train_dataset.x_data_total.shape[2]
seq_len = src_train_dataset.x_data_total.shape[1]
input_dim = (freq_size, seq_len)

model = time_freq_x_vec_DANN(input_dim=input_dim,
                             tdnn_embedding_size=EMBEDDING_SIZE,
                             triplet_output_size=LP_OUTPUT_SIZE,
                             pair_output_size=DC_OUTPUT_SIZE)
if cuda:
    model.cuda()

# 损失函数
LP_loss_triplet = OnlineTripletLoss(TRIPLET_MARGIN, SemihardNegativeTripletSelector(TRIPLET_MARGIN))  # 分类损失：三元损失
DC_loss_domain = torch.nn.NLLLoss()  # 域损失：常规损失
DC_loss_pair = PairWiseLoss(PAIR_MARGIN, KnockPointPairSelector(PAIR_MARGIN))  # 域损失：成对损失
loss_fn = (LP_loss_triplet, DC_loss_pair)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=OFF_INITIAL_LR, weight_decay=OFF_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, OFF_LR_ADJUST_STEP, gamma=OFF_LR_ADJUST_RATIO, last_epoch=-1)

# (3) Baseline model Training & Testing
# transfer_baseline_fit(
#     train_loader=offline_train_loader,
#     val_loader=offline_val_loader,
#     model=model,
#     loss_fn=loss_fn,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     n_epochs=OFFLINE_EPOCH,
#     cuda=cuda)


# ////////////////////////////////////////////// Fine-tuning & Testing /////////////////////////////////////////////////
from utils.fine_tuning_utils import model_parameter_printing
from model_dann_1_xvec.network import fine_tuned_DANN_triplet_Net

# (1) 加载Baseline Model & Baseline Model Test
model_path = '../results/output_model'
model_list = list(os.listdir(model_path))
model_list.sort(reverse=True)
model_name = model_list[1]
baseline_model = torch.load(os.path.join(model_path, model_name))

# (2) 验证基线模型
support_set = [(src_train_dataset.train_data, src_train_dataset.train_label),
               (src_val_dataset.val_data, src_val_dataset.val_label),
               (tgt_train_dataset.train_data, tgt_train_dataset.train_label),
               (tgt_val_dataset.val_data, tgt_val_dataset.val_label),
               ]
query_set = [torch.utils.data.DataLoader(src_train_dataset, batch_size=len(src_train_dataset)),
             torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset)),
             torch.utils.data.DataLoader(tgt_train_dataset, batch_size=len(tgt_train_dataset)),
             torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset)),
             ]
experiment_index = (3, 1)

tgt_sample, src_sample = pair_wise_dataset.__getitem__(random.randint(0, len(pair_wise_dataset)-1))
spec_diff = src_sample - tgt_sample

mean_vec, label_set = support_mean_vec_generation(model=baseline_model,
                                                  support_set=support_set[experiment_index[0]],
                                                  cuda=cuda,
                                                  spec_diff=[])

accu, cm = val_epoch(query_set[experiment_index[1]], baseline_model, mean_vec, label_set, cuda)

plot_confusion_matrix(cm=cm,
                      save_path=os.path.join('../results', 'output_confusion_matrix', model_name + '_experiment' + str(experiment_index) + '.png'),
                      classes=[str(i) for i in label_set])

print('\nBaseline model validation accuracy: %0.2f %%' % accu)

# (3) 修改网络结构
fine_tuned_model = fine_tuned_DANN_triplet_Net(baseline_model, len(SUPPORT_SET_LABEL))

# (4) 固定网络参数
fixed_module = ['feature_extractor', 'domain_classifier']
# fixed_module = ['domain_classifier']
for name, param in fine_tuned_model.named_parameters():
    net_module = name.split('.')[0]
    if net_module in fixed_module:
        param.requires_grad = False

model_parameter_printing(fine_tuned_model)  # 打印网络参数

# (5) Re-initialize Loss_func and Optimizer
loss_fn = torch.nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ON_INITIAL_LR, weight_decay=ON_WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, ON_LR_ADJUST_STEP, gamma=ON_LR_ADJUST_RATIO, last_epoch=-1)

# (6) Fine-tuning & Testing
fine_tuning_fit(
    train_loader=fine_tuning_set_loader,
    test_loader=online_query_loader,
    support_label_set=SUPPORT_SET_LABEL,
    model=fine_tuned_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=ONLINE_EPOCH,
    cuda=cuda
)

# /////////////////////////////////////////////// 可视化观察 ////////////////////////////////////////////////////////////
from utils.embedding_visualization import t_SNE_visualization
t_SNE_visualization(src_train_dataset, tgt_train_dataset, baseline_model, cuda, model_list[0])
