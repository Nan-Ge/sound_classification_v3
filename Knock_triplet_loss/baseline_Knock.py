from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

from data_loader import *
from trainer import fit
from plot import *
from datasets import *

# cuda
cuda = torch.cuda.is_available()

# 加载数据
root_dir = 'E:\\Program\\Acoustic-Expdata-Backend\\Knock_dataset'
domain = 'exp_data'

train_dataset = KnockDataset(root_dir, domain, train=True)
test_dataset = KnockDataset(root_dir, domain, train=False)
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


# 设定Network和训练参数
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 40
log_interval = 50

# 训练baseline
fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# 画图
train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model, cuda)
plot_embeddings(train_embeddings_baseline, train_labels_baseline, n_classes)
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model, cuda)
plot_embeddings(val_embeddings_baseline, val_labels_baseline, n_classes)
