import os
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
from dataset import *
from trainer import *
from torchvision import transforms


######################################## Baseline Training & Testing  ###################################################

### Dataset & DataLoader
root_dir = 'Knock_dataset/fbank_denoised_data'
source_domain = 'exp_data'
target_domain = 'sim_data'
mean, std = 0.1307, 0.3081
support_label_set = {0, 1, 2, 3, 4, 5, 6}

cuda = torch.cuda.is_available()
kwargs = {}

# 常规的Train-Test split：训练集的所有label与测试集所有label相同
# source_train_dataset = KnockDataset(root_dir, source_domain, train=True)
# target_train_dataset = KnockDataset(root_dir, target_domain, train=True)
# domain = (source_domain, target_domain)
# test_dataset = KnockDataset(root_dir, domain, train=False)

# Train-Test-Support split for triplet-loss：测试集中的label完全没有在训练集中出现过
source_train_dataset = KnockDataset_train(root_dir, source_domain, support_label_set)
target_train_dataset = KnockDataset_train(root_dir, target_domain, support_label_set)
test_dataset = KnockDataset_test(root_dir, source_domain, support_label_set)
support_dataset = KnockDataset_test(root_dir, target_domain, support_label_set)
pair_wise_dataset = KnockDataset_pair(root_dir, support_label_set=support_label_set)

# Baseline DataLoader
# batch_size = 32
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
#
# train_loader = torch.utils.data.DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

# Triplet-Loss DataLoader
'''
Online pair selection: We'll create mini batches by sampling labels that will be present in the mini batch and number
of examples from each class, 生成mini-batch的大小为 n_classes * n_samples_per_class
'''
train_n_classes = source_train_dataset.n_classes
test_n_classes = test_dataset.n_classes
n_samples_per_class = 8
# Source Train
source_train_batch_sampler = BalancedBatchSampler(source_train_dataset.train_labels, n_classes=train_n_classes, n_samples=n_samples_per_class)
source_online_train_loader = torch.utils.data.DataLoader(source_train_dataset, batch_sampler=source_train_batch_sampler, **kwargs)
# Target Train
target_train_batch_sampler = BalancedBatchSampler(target_train_dataset.train_labels, n_classes=train_n_classes, n_samples=n_samples_per_class)
target_online_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_sampler=target_train_batch_sampler, **kwargs)
# Test
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
# Pair-wise Train
pair_wise_batch_size = 30
pair_wise_train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_size=pair_wise_batch_size)

online_train_loader = (source_online_train_loader, target_online_train_loader, pair_wise_train_loader)


### 网络、损失函数、优化器
from network import DANN_triplet_Net, X_vector
from Knock_triplet_loss.losses import OnlineTripletLoss, PairWiseLoss
from Knock_triplet_loss.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, KnockPointPairSelector # Strategies for selecting triplets within a minibatch
from Knock_triplet_loss.metrics import AverageNonzeroTripletsMetric


# 网络模型
freq_size = source_train_dataset.x_data_total.shape[2]
seq_len = source_train_dataset.x_data_total.shape[1]
input_dim = (freq_size, seq_len)

tdnn_embedding_size = 128
triplet_output_size = 64
pair_output_size = 64
model = DANN_triplet_Net(input_dim=input_dim, tdnn_embedding_size=tdnn_embedding_size, triplet_output_size=triplet_output_size, pair_output_size=pair_output_size)
if cuda:
    model.cuda()

# 损失函数
triplet_margin = 1.0
pair_margin = 1e-3
LP_loss_triplet = OnlineTripletLoss(triplet_margin, SemihardNegativeTripletSelector(triplet_margin))  # 分类损失：三元损失
DC_loss_domain = torch.nn.NLLLoss()  # 域损失：常规损失
DC_loss_pair = PairWiseLoss(pair_margin, KnockPointPairSelector(pair_margin))  # 域损失：成对损失
loss_fn = (LP_loss_triplet, DC_loss_pair)

# 优化器
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 1

### Baseline model Training & Testing
# baseline_fit(
#     train_loader=online_train_loader,
#     val_loader=online_test_loader,
#     support_set=support_dataset,
#     model=model,
#     loss_fn=loss_fn,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     n_epochs=n_epochs,
#     cuda=cuda,
#     log_interval=log_interval,
#     metrics=[AverageNonzeroTripletsMetric()])

############################################# Fine-tuning & Testing ####################################################
from utils.fine_tuning_utils import model_parameter_printing
from network import fine_tuned_DANN_triplet_Net

### Dataset & DataLoader
batch_size = 20
fine_tuning_set_loader = torch.utils.data.DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

### 加载Baseline Model & Baseline Model Test
model_path = './output_model'
model_list = list(os.listdir(model_path))
model_list.sort(reverse=True)
baseline_model = torch.load(os.path.join(model_path, model_list[0]))

support_set_mean_vec, _ = support_mean_vec_generation(baseline_model, support_dataset, cuda)  # 生成support set mean vector
accu = test_epoch(online_test_loader, baseline_model, support_set_mean_vec, support_label_set, cuda)
print('\nBaseline Model test accuracy: %f' % accu)

### 修改网络结构
fine_tuned_model = fine_tuned_DANN_triplet_Net(baseline_model, len(support_label_set))

### 固定网络参数
fixed_module = ['feature_extractor', 'domain_classifier']
# fixed_module = ['domain_classifier']
for name, param in fine_tuned_model.named_parameters():
    net_module = name.split('.')[0]
    if net_module in fixed_module:
        param.requires_grad = False

model_parameter_printing(fine_tuned_model)  # 打印网络参数

### Re-initialize Loss_func and Optimizer
n_epochs = 200
lr = 1e-3
loss_fn = torch.nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)

### X-vector test model
# x_vec = X_vector(input_dim=input_dim, tdnn_embedding_size=tdnn_embedding_size, n_class=len(support_label_set))

### Fine-tuning & Testing
fine_tuning_fit(
    train_loader=fine_tuning_set_loader,
    test_loader=online_test_loader,
    support_set_mean_vec=[],
    model=fine_tuned_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=n_epochs,
    cuda=cuda
)

################################################## 可视化观察 ############################################################
from utils.embedding_visualization import t_SNE_visualization

t_SNE_visualization(source_train_dataset, target_train_dataset, baseline_model, cuda, model_list[0])


# train_embeddings_online_triplet, train_labels_online_triplet = extract_embeddings(train_loader, model, cuda)
# plot_embeddings(train_embeddings_online_triplet, train_labels_online_triplet, n_classes)
# val_embeddings_online_triplet, val_labels_online_triplet = extract_embeddings(test_loader, model, cuda)
# plot_embeddings(val_embeddings_online_triplet, val_labels_online_triplet, n_classes)

# Accuracy
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
# neigh = KNeighborsClassifier(n_neighbors=1)
# neigh.fit(train_embeddings_online_triplet, train_labels_online_triplet)
# outputs = neigh.predict(val_embeddings_online_triplet)
# acc = accuracy_score(val_labels_online_triplet, outputs)
# print('Accuracy:', acc)
