import os

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch


def t_SNE_visualization(source_dataset, target_dataset, model, cuda, model_name):
    ### 读取数据
    s_x = source_dataset.train_data
    s_y = np.array(source_dataset.train_label, dtype=np.int32)

    t_x = target_dataset.train_data
    t_y = np.array(target_dataset.train_label, dtype=np.int32)

    t_sne = TSNE(n_components=2, init='pca', random_state=100)  # 定义t-SNE可视化

    ### 前向传播
    with torch.no_grad():
        model.eval()

        if cuda:
            model.cuda()
            s_x = s_x.cuda()
            t_x = t_x.cuda()

        s_dc_output, s_lp_output, s_x_embed = model(input_data=s_x, alpha=0)
        t_dc_output, t_lp_output, t_x_embed = model(input_data=t_x, alpha=0)

    ### 使用DC和LP的concat作为可视化的源数据
    s_x_embed = torch.cat((s_dc_output, s_lp_output), dim=1)
    t_x_embed = torch.cat((t_dc_output, t_lp_output), dim=1)

    ### 直接使用embed作为可视化的源数据
    s_x_embed = np.array(s_x_embed.cpu())
    t_x_embed = np.array(t_x_embed.cpu())

    total_label = np.hstack((s_y, t_y))
    total_embed = np.vstack((s_x_embed, t_x_embed))
    total_tsne = t_sne.fit_transform(total_embed)

    ### 归一化
    tsne_min, tsne_max = total_tsne.min(0), total_tsne.max(0)
    tsne_norm = (total_tsne - tsne_min) / (tsne_max - tsne_min)

    ### plot
    marker_list = ['.', 'x']
    plt.figure(figsize=(5, 5), dpi=600)

    # for i in range(s_x_embed.shape[0]):
    #     plt.text(tsne_norm[i, 0], tsne_norm[i, 1], str(total_label[i]), color=plt.cm.Set3(total_label[i] - np.min(total_label)), fontdict={'weight': 'light', 'size': 6})
    # for i in range(s_x_embed.shape[0], s_x_embed.shape[0] + t_x_embed.shape[0]):
    #     plt.text(tsne_norm[i, 0], tsne_norm[i, 1], str(total_label[i]), color=plt.cm.Set3(total_label[i] - np.min(total_label)), fontdict={'weight': 'light', 'size': 6})

    for i in range(s_x_embed.shape[0]):
        plt.scatter(tsne_norm[i, 0], tsne_norm[i, 1], color=plt.cm.Set3(total_label[i] - np.min(total_label)), marker=marker_list[0], s=25)
    for i in range(s_x_embed.shape[0], s_x_embed.shape[0] + t_x_embed.shape[0]):
        plt.scatter(tsne_norm[i, 0], tsne_norm[i, 1], color=plt.cm.Set3(total_label[i] - np.min(total_label)), marker=marker_list[1], s=25)

    plt.savefig(os.path.join('output_training_log', model_name + 't_SNE_embedding.png'))
    print('t_SNE have been saved:' + model_name)