import torch
import numpy as np
import matplotlib.pyplot as plt


dataset_classes = ['changmuban', 'duanmuban', 'lvban', 'tieban', 'xiaozhuodi', 'xiaozhuoding',
                   'xiaozhuozuo', 'yinxiangbei', 'yinxiangdi', 'yinxiangding', 'yinxiangyou',
                   'yklban', 'zhuogeban']
dataset_classes2 = ['changmuban', 'duanmuban', 'lvban', 'tieban', 'xiaozhuoding',
                   'yinxiangdi', 'yinxiangding', 'yklban', 'zhuogeban']
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#66ccff', '#ee0000',
          '#9999ff', '#ffff00']


def plot_embeddings(embeddings, targets, n_classes, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(dataset_classes2)
    plt.show()


def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k: k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k: k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
