from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
from model_dann_1_xvec.dataset import *
from model_dann_1_xvec.online_trainer import *
from model_dann_1_xvec.network import X_vector

from get_train_test import *
from config_2 import *
from baseline_trainer import *


def train_base_model(train_dataset, test_dataset):
    cuda = torch.cuda.is_available()
    n_classes = train_dataset.n_classes

    x_vec_train_batch_size = 50
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(x_vec_train_batch_size), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    # (2) 网络、损失函数、优化器
    seq_len = train_dataset.knock_data.shape[1]
    freq_size = train_dataset.knock_data.shape[2]
    input_dim = (freq_size, seq_len)

    # 网络模型
    model = X_vector(input_dim=input_dim[0], tdnn_embedding_size=EMBEDDING_SIZE, n_class=n_classes)
    model.cuda()

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=OFF_INITIAL_LR, weight_decay=OFF_WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, OFF_LR_ADJUST_STEP, gamma=OFF_LR_ADJUST_RATIO, last_epoch=-1)

    # 训练并查看效果
    non_transfer_base_line_fit(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=OFFLINE_EPOCH,
        cuda=cuda
    )


if __name__ == '__main__':
    # (1) 数据集
    name = 'exp'
    target_path = 'fbank_dnoised_data'
    obj_list = ObjList(name)
    dataset = load_feature_data(obj_list, target_path)
    train_dataset, test_dataset = get_train_test_dataset_3(dataset, 20)
    print('Loading data finished.')

    train_base_model(train_dataset, test_dataset)
