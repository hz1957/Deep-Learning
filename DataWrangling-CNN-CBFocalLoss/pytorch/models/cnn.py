import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and no padding                           #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 13 * 13, out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.features(x)
        outs = self.flatten(outs)
        outs = self.linear(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
