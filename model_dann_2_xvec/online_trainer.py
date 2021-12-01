import torch
import torch.nn as nn

import numpy as np
import sys
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix


def netFT_fit(train_loader, test_loader, support_label_set, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_save_name_1 = 'xvec_dann_orig_FineTuning' + exp_time + '.txt'
    print('\n--------- Start fine-tuning at:' + exp_time + '---------')

    for epoch in range(n_epochs):
        scheduler.step()
        # Train stage
        netFT_train_epoch(train_loader=train_loader, model=model,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          epoch=epoch,
                          cuda=cuda,
                          support_label_set=support_label_set)
        # Test stage
        test_accu = netFT_test_epoch(test_loader=test_loader,
                                     model=model,
                                     cuda=cuda,
                                     support_label_set=support_label_set)
        print(', Test accuracy: %f for %d / %d' % (test_accu, epoch + 1, n_epochs))


def netFT_train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, support_label_set):
    model.train()
    len_dataloader = len(train_loader)

    for index, item in enumerate(train_loader):
        sound, label = item
        sound = sound.squeeze()  # 删除channel dimension

        # 调整label的值，以进行NLLLoss的计算
        for i, label_ in enumerate(label):
            label[i] = torch.tensor(support_label_set.index(label[i]))

        if cuda:
            model.cuda()
            sound = sound.cuda()
            label = label.cuda()

        model.zero_grad()
        pred, _ = model(sound)
        # _, _, pred = model(sound)
        err_label = loss_fn(pred, label)
        err_label.backward()
        optimizer.step()

        # 打印信息
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_label: %f' %
                         (epoch,
                          index + 1,
                          len_dataloader,
                          err_label.data.cpu().numpy()))
        sys.stdout.flush()
        # print('\r epoch: %d, [iter: %d / all %d], err_label: %f' %
        #                  (epoch,
        #                   index + 1,
        #                   len_dataloader,
        #                   err_label.data.cpu().numpy()))
    return err_label


def netFT_test_epoch(test_loader, model, cuda, support_label_set):
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        model.eval()

        test_iter = iter(test_loader)
        test_sound, test_label = test_iter.next()

        # 调整label的值，以进行NLLLoss的计算
        for i, label_ in enumerate(test_label):
            test_label[i] = torch.tensor(support_label_set.index(test_label[i]))

        test_sound = test_sound.squeeze()  # 删除channel dimension
        test_label = np.array(test_label.cpu())

        if cuda:
            test_sound = test_sound.cuda()

        pred, _ = model(test_sound)
        # _, _, pred = model(test_sound)
        pred_label = np.argmax(np.array(pred.cpu()), axis=1)

        n_correct += sum(test_label == pred_label)
        n_total += len(test_label)
        accu = n_correct / n_total

        confusion_mat = confusion_matrix(test_label, pred_label)

    return accu, confusion_mat
