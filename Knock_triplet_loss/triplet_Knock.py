from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

from Knock_triplet_loss.trainer import fit
from Knock_triplet_loss.plot import *
from Knock_triplet_loss.datasets_Knock import *

# cuda
cuda = torch.cuda.is_available()

# 加载数据
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
# 加载数据
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, drop_last=True, **kwargs)

# Siamese
# 加载数据
triplet_train_dataset = TripletKnock(train_dataset)
triplet_test_dataset = TripletKnock(test_dataset)

batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# 设定Network和训练参数
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

# 训练Triplet
fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval)

# 画图
train_embeddings_triplet, train_labels_triplet = extract_embeddings(train_loader, model, cuda)
plot_embeddings(train_embeddings_triplet, train_labels_triplet, n_classes)
val_embeddings_triplet, val_labels_triplet = extract_embeddings(test_loader, model, cuda)
plot_embeddings(val_embeddings_triplet, val_labels_triplet, n_classes)

# Accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_embeddings_triplet, train_labels_triplet)
outputs = neigh.predict(val_embeddings_triplet)
acc = accuracy_score(val_labels_triplet, outputs)
print('Accuracy:', acc)
