from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

from data_loader import *
from trainer import fit
from plot import *
from datasets_Knock import *

# cuda
cuda = torch.cuda.is_available()

# 加载数据
from torchvision import transforms

root_dir = 'E:\\Program\\Acoustic-Expdata-Backend\\Knock_dataset'
domain = 'exp_data'

mean, std = 0.1307, 0.3081

train_dataset = KnockDataset(root_dir, domain, train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = KnockDataset(root_dir, domain, train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
n_classes = train_dataset.n_classes

# Baseline
# 加载数据
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, drop_last=True, **kwargs)

# Online pair selection
# 加载数据
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes, n_samples=25)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# 设定Network和训练参数
from networks import EmbeddingNet
from losses import OnlineContrastiveLoss
from utils.loss_utils import HardNegativePairSelector # Strategies for selecting pairs within a minibatch

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

# 训练Online pair selection
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval)

# 画图
train_embeddings_online_siamese, train_labels_online_siamese = extract_embeddings(train_loader, model, cuda)
plot_embeddings(train_embeddings_online_siamese, train_labels_online_siamese, n_classes)
val_embeddings_online_siamese, val_labels_online_siamese = extract_embeddings(test_loader, model, cuda)
plot_embeddings(val_embeddings_online_siamese, val_labels_online_siamese, n_classes)

# Accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_embeddings_online_siamese, train_labels_online_siamese)
outputs = neigh.predict(val_embeddings_online_siamese)
acc = accuracy_score(val_labels_online_siamese, outputs)
print('Accuracy:', acc)
