import torch
import numpy as np


# 计算两个向量组之间的欧式距离
def cal_euclidean(embedding, mean_vec):
    ecu_sim = np.empty(shape=(0, mean_vec.shape[0]))
    for i in range(0, embedding.shape[0]):
        single_sim = np.square(mean_vec-embedding[i, :]).sum(1)[np.newaxis, :]
        ecu_sim = np.vstack((ecu_sim, single_sim))
    return ecu_sim


# generate the mean vectors of the support set
def support_mean_vec_generation(model, support_set, cuda, spec_diff=[]):
    support_mean_vec = []
    spt_x = support_set[0]
    spt_y = support_set[1]
    spt_label_set = list(set(list(np.array(spt_y, dtype=np.int32))))
    spt_label_set.sort()

    if isinstance(spec_diff, torch.Tensor):
        spt_x = spt_x + spec_diff

    with torch.no_grad():
        model.eval()
        if cuda:
            model.cuda()
            spt_x = spt_x.cuda()
        spt_embed, _ = model(input_data=spt_x, alpha=0)
        for label in iter(spt_label_set):
            temp = spt_embed[spt_y == label]
            temp_avg = np.mean(np.array(temp.cpu()), axis=0)
            support_mean_vec.append(temp_avg)
        support_mean_vec = np.array(support_mean_vec)
    return support_mean_vec, spt_label_set