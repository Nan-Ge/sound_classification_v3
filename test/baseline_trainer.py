import torch

import numpy as np
import sys
import time
from sklearn.metrics import confusion_matrix


# --------------------------------------------- 非迁移学习训练 -----------------------------------------------------------
# Train an epoch on the x-vector
def train_epoch_x_vec(train_loader, model, loss_fn, optimizer, cuda, epoch):
    train_iter = iter(train_loader)

    err_softmax = None
    for i in range(len(train_loader)):
        model.zero_grad()

        train_data, train_label = next(train_iter)

        if cuda:
            train_data = train_data.cuda()
            train_label = train_label.cuda()

        train_label = train_label - 1

        # 前向传播
        _, _, pred = model(input_data=train_data)

        # Softmax Loss
        err_softmax = loss_fn(pred, train_label)

        # 反向传播
        err_softmax.backward()
        optimizer.step()

        # Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_softmax: %f' %
                         (epoch, i + 1, len(train_loader), err_softmax.data.cpu().numpy()))
        sys.stdout.flush()

    return err_softmax


# Validate the x-vector
def test_epoch_x_vec(test_loader, model, cuda):
    n_total = 0
    n_correct = 0

    test_iter = iter(test_loader)

    with torch.no_grad():
        model.eval()
        val_data, val_label = next(test_iter)
        val_label = np.array(val_label.cpu())
        if cuda:
            val_data = val_data.cuda()
        val_label = val_label - 1

        # 前向传播
        _, _, pred_logit = model(input_data=val_data)

        # Inference
        pred_label = np.argmax(pred_logit.detach().cpu().numpy(), axis=1)

        # Statistic
        n_correct += sum(val_label == pred_label)
        n_total += len(val_label)
        accu = n_correct / n_total

        # Confusion Matrix
        confusion_mat = confusion_matrix(val_label, pred_label)

        return accu, confusion_mat


def non_transfer_base_line_fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')
    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train Stage
        err_softmax = train_epoch_x_vec(train_loader=train_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        cuda=cuda,
                                        epoch=epoch)

        # Validation Stage
        accu, cm = test_epoch_x_vec(
            test_loader=test_loader,
            model=model,
            cuda=cuda
        )
        print(', validation accuracy of offline x-vector training: %.2f %% for %d / %d' % (
        accu * 100, epoch + 1, n_epochs))

    # model_name = 'x_vector_model_' + exp_time + '_' + str(format(accu, '.2f')) + '.pkl'
    # model_save_path = os.path.join('results/output_model', model_name)
    # torch.save(model, model_save_path)
    # print('\nBaseline Model saved as:', model_save_path)
    #
    # plot_confusion_matrix(cm=cm, savename=model_name + '-confusion_matrix.png', classes=[str(i) for i in range(0,43)])
