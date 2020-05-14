import numpy as np
import math
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


def binary_cross_entropy(model,
                         logits, labels):

    labels = labels.float().view(-1)

    bce_loss = nn.BCELoss(reduction='mean')
    bce = bce_loss(logits.view(-1), labels)

    return bce


def multi_binary_cross_entropy(model,
                               logits, labels,
                               label_masks,
                               label_weights):

    label_weights = label_weights.view(-1)
    label_masks = label_masks.view(-1)
    labels = labels.float().view(-1)

    bce_loss = nn.BCELoss(reduction='none')
    bce = bce_loss(logits.view(-1), labels)
    recall_weights = (labels*label_weights + (1-labels))
    bce = bce*label_masks*recall_weights  # *((1-labels)*label_weights + labels)

    return bce.mean()


def DSC(model,
        n_fold,
        logits, labels,
        label_masks,
        label_weights):

    N = labels.size(0)
    labels = labels.view(N, n_fold, -1)
    labels = labels.view(N*n_fold, -1)
    label_masks = label_masks.view(N, n_fold, -1)
    label_masks = label_masks.view(N*n_fold, -1)

    gamma = 1

    p = (1-logits)*logits*label_masks
    y = labels*label_masks

    # print(logits)
    # print(y)

    loss = 1.0-(((p*y).sum(0)+gamma)/(p.sum(0)+y.sum(0)+gamma))

    loss = loss  # *label_weights[0]

    loss = loss.mean()

    return loss


def DSC_(model,
         n_fold,
         logits, labels,
         label_masks,
         label_weights):

    N = labels.size(0)
    labels = labels.view(N, n_fold, -1)
    labels = labels.view(N*n_fold, -1)
    label_masks = label_masks.view(N, n_fold, -1)
    label_masks = label_masks.view(N*n_fold, -1)

    gamma = 1.0

    p = (1.0-logits)*logits*label_masks
    y = labels*label_masks

    # print(logits)
    # print(y)

    loss = 1.0-(((p*y)+gamma)/(p+y+gamma))

    # loss = loss  # *label_weights[0]

    loss = loss.mean()

    return loss
