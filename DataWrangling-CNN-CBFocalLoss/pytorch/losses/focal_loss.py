import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = (1.0 - beta) / np.array(1.0 - np.power(beta, cls_num_list))
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

    per_cls_weights = torch.FloatTensor(per_cls_weights)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma * ce).mean()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
