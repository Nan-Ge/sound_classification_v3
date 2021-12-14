import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from config import *

import os

from sklearn.utils import shuffle


class TripletKnock(Dataset):

    def __init__(self, knock_dataset):
        self.knock_dataset = knock_dataset
        self.train = self.knock_dataset.train

        self.transform = self.knock_dataset.transform

        if self.train:
            self.train_labels = self.knock_dataset.knock_labels
            self.train_data = self.knock_dataset.knock_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.val_labels = self.knock_dataset.knock_labels
            self.val_data = self.knock_dataset.knock_data
            # generate fixed triplets for testing
            self.labels_set = set(self.val_labels.numpy())
            self.label_to_indices = {label: np.where(self.val_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.val_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.val_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.val_data))]
            self.val_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.val_data[self.val_triplets[index][0]]
            img2 = self.val_data[self.val_triplets[index][1]]
            img3 = self.val_data[self.val_triplets[index][2]]
        #
        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.knock_dataset)
