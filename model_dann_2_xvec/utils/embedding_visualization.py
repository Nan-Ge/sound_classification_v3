import os

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch


def tsne_plot(src_dataset, tgt_dataset, model, cuda, model_name):
    # 读取数据
    src_x = src_dataset[0]
    src_y = np.array(src_dataset[1], dtype=np.int32)

    tgt_x = tgt_dataset[0]
    tgt_y = np.array(tgt_dataset[1], dtype=np.int32)

    t_sne = TSNE(n_components=2, init='pca', random_state=100)  # 定义t-SNE可视化

    # 前向传播
    with torch.no_grad():
        model.eval()

        if cuda:
            model.cuda()
            src_x = src_x.cuda()
            tgt_x = tgt_x.cuda()

        src_embed, _, _ = model(input_data=src_x, alpha=0)
        tgt_embed, _, _ = model(input_data=tgt_x, alpha=0)

    # 直接使用embed作为可视化的源数据
    src_embed = np.array(src_embed.cpu())
    tgt_embed = np.array(tgt_embed.cpu())

    total_label = np.hstack((src_y, tgt_y))
    total_embed = np.vstack((src_embed, tgt_embed))

    total_tsne = t_sne.fit_transform(total_embed)

    # 归一化
    tsne_min, tsne_max = total_tsne.min(0), total_tsne.max(0)
    tsne_norm = (total_tsne - tsne_min) / (tsne_max - tsne_min)

    # plot
    marker_list = ['.', 'x']
    plt.figure(figsize=(8, 8), dpi=600)

    for i in range(src_embed.shape[0]):
        plt.text(
            tsne_norm[i, 0], tsne_norm[i, 1], str(total_label[i]),
            color='b', fontdict={'weight': 'light', 'size': 6})
    for i in range(src_embed.shape[0], src_embed.shape[0] + tgt_embed.shape[0]):
        plt.text(
            tsne_norm[i, 0], tsne_norm[i, 1], str(total_label[i]),
            color='r', fontdict={'weight': 'light', 'size': 6})

    plt.savefig(os.path.join('../results', 'output_embed_visualization', model_name + '_' + 't_SNE_embedding.png'))
    print('t_SNE have been saved:' + model_name)