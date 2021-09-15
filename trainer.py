import torch
import torch.nn as nn

import numpy as np
import sys
import os
import time
from sklearn.metrics.pairwise import cosine_similarity


######################################## Baseline Traning & Testing  ###################################################
def baseline_fit(train_loader, val_loader, support_set, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[], start_epoch=0):
    exp_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_save_name_1 = 'DANN-Triplet-Net_without_FineTuning_' + exp_time + '.txt'
    print('\n--------- Start baseline model training at:' + exp_time + '---------')
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        ### Train stage
        # err_s_label, err_s_domain, err_t_domain, total_iter = train_epoch(train_loader, model, loss_fn, optimizer, cuda, metrics, epoch, n_epochs)
        train_epoch_triplet_pair(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs)

        ### Test stage
        support_set_mean_vecs, support_label_set = support_mean_vec_generation(model, support_set, cuda)  # 生成support set mean vector
        accu = test_epoch(val_loader, model, support_set_mean_vecs, support_label_set, cuda)
        print(', Test accuracy: %f for %d / %d' % (accu, epoch+1, n_epochs))

        # with open(os.path.join('output_result', log_save_name_1), 'a') as f:
        #     train_output = '\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' % (
        #     epoch, total_iter, total_iter, err_s_label, err_s_domain, err_t_domain)
        #     f.write(train_output)
        #     test_output = ', Test accuracy: %f for %d / %d' % (accu, epoch + 1, n_epochs)
        #     f.write(test_output)

    model_name = 'DANN_triplet_baseline_model_' + exp_time + '_' + str(accu) +'.pkl'
    model_save_path = os.path.join('output_model', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)


### Pair-wise loss for label predictor
def train_epoch_triplet_pair(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs):
    model.train()

    LP_source_train_loader = train_loader[0]
    LP_target_train_loader = train_loader[1]
    DC_pair_train_loader = train_loader[2]

    len_dataloader = max(len(DC_pair_train_loader), len(LP_source_train_loader), len(LP_target_train_loader))

    LP_source_iter = iter(LP_source_train_loader)
    LP_target_iter = iter(LP_target_train_loader)
    DC_iter = iter(DC_pair_train_loader)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / (n_epochs * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = alpha * 1000

        model.zero_grad()

        ##### Label Predictor Forwarding
        ### Source Domain
        data_source = LP_source_iter.next()
        s_sound, s_label = data_source
        s_sound = s_sound.squeeze()  # 删除channel dimension

        if cuda:
            s_sound = s_sound.cuda()
            s_label = s_label.cuda()

        # 前向传播
        s_class_output, _, _ = model(input_data=s_sound, alpha=alpha)

        # triplet-loss
        if type(s_class_output) not in (tuple, list):
            s_class_output = (s_class_output,)
        s_triple_loss_inputs = s_class_output
        if s_label is not None:
            s_label = (s_label,)
            s_triple_loss_inputs += s_label
        err_s_label, _ = loss_fn[0](*s_triple_loss_inputs)

        ### Target Domain
        data_target = LP_target_iter.next()
        t_sound, t_label = data_target
        t_sound = t_sound.squeeze()  # 删除channel dimension

        if cuda:
            t_sound = t_sound.cuda()
            t_label = t_label.cuda()

        # 前向传播
        t_class_output, _, _ = model(input_data=t_sound, alpha=alpha)

        # triplet-loss
        if type(t_class_output) not in (tuple, list):
            t_class_output = (t_class_output,)
        t_triple_loss_inputs = t_class_output
        if t_label is not None:
            t_label = (t_label,)
            t_triple_loss_inputs += t_label
        err_t_label, _ = loss_fn[0](*t_triple_loss_inputs)

        ##### Domain Classifier Forwarding
        try:
            data_pair = DC_iter.next()
        except StopIteration:
            DC_iter = iter(DC_pair_train_loader)
            data_pair = DC_iter.next()

        exp_sound, sim_sound = data_pair
        exp_sound = exp_sound.squeeze()
        sim_sound = sim_sound.squeeze()

        if cuda:
            exp_sound = exp_sound.cuda()
            sim_sound = sim_sound.cuda()

        # 前向传播
        _, exp_embedding, _ = model(input_data=exp_sound, alpha=alpha)
        _, sim_embedding, _ = model(input_data=sim_sound, alpha=alpha)

        # pair-wise loss
        pair_wise_loss_inputs = (exp_embedding, sim_embedding)
        err_domain = loss_fn[1](*pair_wise_loss_inputs)

        ##### Totoal loss & Backward
        err_total = err_s_label + err_t_label + err_domain
        err_total.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_t_label: %f, err_domain: %f' %
                         (epoch, i + 1,
                          len_dataloader,
                          err_s_label.data.cpu().numpy(),
                          err_t_label.data.cpu().numpy(),
                          err_domain.data.cpu().numpy()))
        sys.stdout.flush()


### softmax loss for label predictor
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, metrics, epoch, n_epochs):
    for metric in metrics:
        metric.reset()

    model.train()

    source_train_loader = train_loader[0]
    target_train_loader = train_loader[1]
    len_dataloader = max(len(source_train_loader), len(target_train_loader))
    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / (n_epochs * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        ### training model using source data
        # 加载Source Data
        data_source = source_iter.next()
        s_sound, s_label = data_source
        s_sound = s_sound.squeeze()  # 删除channel dimension
        batch_size = len(s_label)
        s_domain_label = torch.zeros(batch_size).long()
        s_label = s_label if len(s_label) > 0 else None
        if cuda:
            s_sound = s_sound.cuda()
            s_label = s_label.cuda()
            s_domain_label = s_domain_label.cuda()
        # 前向传播
        model.zero_grad()
        s_class_output, s_domain_output, _ = model(input_data=s_sound, alpha=alpha)
        # triplet-loss
        if type(s_class_output) not in (tuple, list):
            s_class_output = (s_class_output,)
        s_triple_loss_inputs = s_class_output
        if s_label is not None:
            s_label = (s_label,)
            s_triple_loss_inputs += s_label
        s_triplet_loss_outputs, _ = loss_fn[0](*s_triple_loss_inputs)
        err_s_label = s_triplet_loss_outputs
        # domain loss
        err_s_domain = loss_fn[1](s_domain_output, s_domain_label)

        ### training model using target data
        # 加载Target Data
        data_target = target_iter.next()
        t_sound, t_label = data_target
        t_sound = t_sound.squeeze()  # 删除channel dimension
        batch_size = len(t_sound)
        t_domain_label = torch.ones(batch_size).long()
        if cuda:
            t_sound = t_sound.cuda()
            t_label = t_label.cuda()
            t_domain_label = t_domain_label.cuda()
        # 前向传播
        t_class_output, t_domain_output, _ = model(input_data=t_sound, alpha=alpha)
        # triplet-loss
        if type(t_class_output) not in (tuple, list):
            t_class_output = (t_class_output,)
        t_triple_loss_inputs = t_class_output
        if t_label is not None:
            t_label = (t_label,)
            t_triple_loss_inputs += t_label
        t_triplet_loss_outputs, _ = loss_fn[0](*t_triple_loss_inputs)
        err_t_label = t_triplet_loss_outputs
        # domain loss
        err_t_domain = loss_fn[1](t_domain_output, t_domain_label)

        ### 总损失
        err_total = err_s_domain + err_s_label + err_t_domain + err_t_label
        err_total.backward()
        optimizer.step()

        # 打印信息
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_label: %f, err_t_domain: %f' %
                         (epoch, i + 1,
                          len_dataloader,
                          err_s_label.data.cpu().numpy(),
                          err_s_domain.data.cpu().numpy(),
                          err_t_label.data.cpu().numpy(),
                          err_t_domain.data.cpu().item()))
        sys.stdout.flush()

    return err_s_label.data.cpu().numpy(), err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), len_dataloader


def support_mean_vec_generation(model, support_set, cuda):
    support_mean_vecs = []
    x_data_support_set = support_set.test_data
    y_data_support_set = support_set.test_labels
    support_label_set = list(set(list(np.array(y_data_support_set, dtype=np.int32))))
    support_label_set.sort()
    with torch.no_grad():
        model.eval()
        if cuda:
            model.cuda()
            x_data_support_set = x_data_support_set.cuda()
        _, _, support_embedding = model(input_data=x_data_support_set, alpha=0)
        for label in iter(support_label_set):
            temp = support_embedding[y_data_support_set == label]
            temp = np.mean(np.array(temp.cpu()), axis=0)
            support_mean_vecs.append(temp)
        support_mean_vecs = np.array(support_mean_vecs)
    return support_mean_vecs, support_label_set


def test_epoch(test_loader, model, support_set_mean_vecs, support_label_set, cuda):
    n_total = 0
    n_correct = 0
    support_label_set = [int(i) for i in iter(support_label_set)]  # 浮点型变为整型

    with torch.no_grad():
        model.eval()
        test_iter = iter(test_loader)
        test_sound, test_label = test_iter.next()
        test_sound = test_sound.squeeze()  # 删除channel dimension
        test_label = np.array(test_label.cpu())

        if cuda:
            test_sound = test_sound.cuda()

        _, _, test_embedding = model(input_data=test_sound, alpha=0)
        test_embedding = np.array(test_embedding.cpu())
        cos_sim = cosine_similarity(test_embedding, support_set_mean_vecs)
        pred_label = np.argmax(cos_sim, axis=1)

        # for i in range(len(pred_label)):
        #     pred_label[i] = support_label_set[pred_label[i]]

        n_correct += sum(test_label == pred_label)
        n_total += len(test_label)
        accu = n_correct / n_total

    return accu


############################################# Fine tuning & Testing ####################################################
def fine_tuning_fit(train_loader, test_loader, support_set_mean_vec, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_save_name_1 = 'DANN-Triplet-Net_FineTuning' + exp_time + '.txt'
    print('\n--------- Start fine-tuning at:' + exp_time + '---------')

    for epoch in range(n_epochs):
        scheduler.step()
        # Train stage
        fine_tuning_epoch(train_loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch, cuda=cuda)
        # Test stage
        test_accu = fine_tuning_test_epoch(test_loader=test_loader, model=model, cuda=cuda)
        print(', Test accuracy: %f for %d / %d' % (test_accu, epoch + 1, n_epochs))


def fine_tuning_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch):
    model.train()
    len_dataloader = len(train_loader)

    for index, item in enumerate(train_loader):
        sound, label = item
        sound = sound.squeeze()  # 删除channel dimension
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
                         (epoch, index + 1,
                          len_dataloader,
                          err_label.data.cpu().numpy()))
        sys.stdout.flush()
        # print('\r epoch: %d, [iter: %d / all %d], err_label: %f' %
        #                  (epoch, index + 1,
        #                   len_dataloader,
        #                   err_label.data.cpu().numpy()))
    return 0


def fine_tuning_test_epoch(test_loader, model, cuda):
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        model.eval()

        test_iter = iter(test_loader)
        test_sound, test_label = test_iter.next()
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

    return accu








