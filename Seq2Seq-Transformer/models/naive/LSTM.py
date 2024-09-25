

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                             #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: the forget gate
        self.Wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate
        self.Wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate
        self.Wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        N, S, F = x.size()

        h_t, c_t = torch.zeros(N, self.hidden_size), torch.zeros(N, self.hidden_size)

        for t in range(S):
            x_t = x[:, t, :]

            i_t = self.sigmoid(torch.matmul(x_t, self.Wii) + torch.matmul(h_t, self.Whi) + self.bi)
            f_t = self.sigmoid(torch.matmul(x_t, self.Wif) + torch.matmul(h_t, self.Whf) + self.bf)
            o_t = self.sigmoid(torch.matmul(x_t, self.Wio) + torch.matmul(h_t, self.Who) + self.bo)
            g_t = self.tanh(torch.matmul(x_t, self.Wig) + torch.matmul(h_t, self.Whg) + self.bg)

            c_t = i_t * g_t + f_t * c_t
            h_t = o_t * self.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
