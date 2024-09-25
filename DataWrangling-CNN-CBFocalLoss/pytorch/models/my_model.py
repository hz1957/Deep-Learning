
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.model = nn.Sequential(
            # 1st Convolutional Layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd Convolutional Layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd Convolutional Layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten
            nn.Flatten(),

            # 1st Linear Layer
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # 2nd Linear Layer (Output Layer)
            nn.Linear(512, 10)
            )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.model(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
