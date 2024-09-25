import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for h_ in range(H_out):
                    for w_ in range(W_out):
                        h_start = h_*self.stride
                        w_start = w_*self.stride
                        x_slice = x[n, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]

                        out[n, c, h_, w_] = np.max(x_slice)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, _, _ = x.shape

        # Initialize the gradient
        self.dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h_ in range(H_out):
                    for w_ in range(W_out):
                        h_start = h_*self.stride
                        w_start = w_*self.stride
                        x_slice = x[n, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
              
                         # Find the indices of the maximum value
                        max_indices = np.unravel_index(np.argmax(x_slice), x_slice.shape)

                        # Update the gradient with respect to the input
                        self.dx[n, c, h_start + max_indices[0], w_start + max_indices[1]] += dout[n, c, h_, w_]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
