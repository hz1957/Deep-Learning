import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, _, H, W = x.shape
        H_ = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_ = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        out = np.zeros((N, self.out_channels, H_, W_))

        for n in range(N):
            for c_ in range(self.out_channels):
                for h_ in range(H_):
                    for w_ in range(W_):
                        # Extract the slice from the padded input
                        h_start = self.stride * h_
                        w_start = self.stride * w_
                        x_slice = x_padded[n, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
        
                        # Perform the convolution operation
                        out[n, c_, h_, w_] = np.sum(x_slice * self.weight[c_, :, :, :]) + self.bias[c_]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, _, H, W = x.shape
        H_ = (self.padding * 2 + H - self.kernel_size) // self.stride + 1
        W_ = (self.padding * 2 + W - self.kernel_size) // self.stride + 1
        x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        self.dx = np.zeros_like(x)
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)
        dx_padded = np.zeros_like(x_padded)

        for n in range(N):
            for c_ in range(self.out_channels):
                # #  compute the gradients with respect to the bias
                self.db[c_] += dout[n, c_, :, :].sum()
                for h_ in range(H_):
                    for w_ in range(W_):
                        h_start = self.stride * h_
                        w_start = self.stride * w_
                        x_slice = x_padded[n, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        #  compute the gradients with respect to the weight
                        self.dw[c_, :, :, :] += x_slice * dout[n, c_, h_, w_]

                        #  compute the gradients with respect to the input 
                        dx_padded[n, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += self.weight[c_, :, :, :] * dout[n, c_, h_, w_]
        
        # remove padding from dx_padded
        self.dx = dx_padded[:, :, self.padding: self.padding+H, self.padding: self.padding+W]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
