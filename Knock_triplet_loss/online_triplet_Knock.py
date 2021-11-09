from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

from Knock_triplet_loss.data_loader import *
from Knock_triplet_loss.trainer import fit
from Knock_triplet_loss.plot import *
from Knock_triplet_loss.datasets_Knock import *

# cuda
cuda = torch.cuda.is_available()

from torchvision import transforms

root_dir = '../Knock_dataset'
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
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

# Online pair selection
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# 生成mini-batch的大小为 n_classes * n_samples_per_class
n_samples_per_class = 10
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=n_samples_per_class)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes, n_samples=n_samples_per_class)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# 设定Network和训练参数
from Knock_triplet_loss.networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils.loss_utils import \
    RandomNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from Knock_triplet_loss.metrics import AverageNonzeroTripletsMetric

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

# 训练Online triplet selection
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

# 画图
train_embeddings_online_triplet, train_labels_online_triplet = extract_embeddings(train_loader, model, cuda)
plot_embeddings(train_embeddings_online_triplet, train_labels_online_triplet, n_classes)
val_embeddings_online_triplet, val_labels_online_triplet = extract_embeddings(test_loader, model, cuda)
plot_embeddings(val_embeddings_online_triplet, val_labels_online_triplet, n_classes)

# Accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_embeddings_online_triplet, train_labels_online_triplet)
outputs = neigh.predict(val_embeddings_online_triplet)
acc = accuracy_score(val_labels_online_triplet, outputs)
print('Accuracy:', acc)
