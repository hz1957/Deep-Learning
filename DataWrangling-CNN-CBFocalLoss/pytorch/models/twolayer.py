
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1),
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x_flattened = x.view(x.size(0), -1)
        out = self.model(x_flattened)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
