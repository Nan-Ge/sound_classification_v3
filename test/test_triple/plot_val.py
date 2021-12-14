import torch
import numpy as np
import matplotlib.pyplot as plt


knock_classes = [str(i) for i in range(43)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(43):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i % 10])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(knock_classes)
    plt.show()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.knock_data), 2))
        # labels = np.zeros(len(dataloader.knock_data))

        images = dataloader.knock_data

        length = len(dataloader.knock_data)
        batch_size = 100
        batch_num = length // batch_size

        for i in range(batch_num):
            if torch.cuda.is_available():
                image = images[i * batch_size: (i + 1) * batch_size].cuda()
            embeddings[i * batch_size: (i + 1) * batch_size] = model.get_embedding(image).data.cpu().numpy()
        labels = dataloader.knock_labels

    return embeddings, labels