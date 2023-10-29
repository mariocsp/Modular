"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torch
from torch import nn


class Model_base(nn.Module):

    def __init__(self, input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        """
        Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
        See the original architecture here: https://poloclub.github.io/cnn-explainer/

        Args:
            input_shape: An integer indicating number of input channels.
            hidden_units: An integer indicating number of hidden units between layers.
            output_shape: An integer indicating number of output units.
        """

        super().__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride  = 2 ))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride  = 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4205,
                      out_features=output_shape)
        )


    def forward(self, x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))