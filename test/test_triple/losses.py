import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class PairWiseLoss(nn.Module):
    def __init__(self, margin, pair_selector):
        super(PairWiseLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, exp_embedding, sim_embedding):
        pairs = self.pair_selector.get_pairs((exp_embedding, sim_embedding), [])
        pair_dist = (exp_embedding[pairs, :] - sim_embedding[pairs, :]).pow(2).sum(1)
        losses = F.relu(pair_dist - self.margin)
        return losses.mean()


class OnlineTripletLoss_Test(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss_Test, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()


class xvec_dann_tripletloss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(xvec_dann_tripletloss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, target, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class xvec_dann_dc_tripletloss(nn.Module):

    def __init__(self):
        super(xvec_dann_dc_tripletloss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, y1, y2, y3, domain_label):
        err_dc1 = self.loss(y1, domain_label)
        err_dc2 = self.loss(y2, domain_label)
        err_dc3 = self.loss(y3, domain_label)
        err_dc = err_dc1 + err_dc2 + err_dc3
        return err_dc * 50
