

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        # Step 1: Forward process
        linear_output = np.dot(X, self.weights['W1'])
        relu_output = self.ReLU(linear_output)

        # Step 2: Calculate loss and accuracy
        probabilities = self.softmax(relu_output)
        loss = self.cross_entropy_loss(probabilities, y)
        accuracy = self.compute_accuracy(probabilities, y)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        N = len(X)
        
        # Step 1: Compute gradient of Cross-Entropy loss with respect to RELU output
        last_dev = probabilities.copy()
        last_dev[np.arange(N), y] -= 1
        last_dev *= 1/N

        # Step 2: Compute gradient of ReLU output with respect to linear output
        relu_dev = self.ReLU_dev(linear_output)

        # Step 3: Compute gradient of linear output with respect to W1
        linear_dev = X

        # Step 4: Compute gradient of loss with respect to W1
        self.gradients['W1'] = np.dot(linear_dev.T, last_dev * relu_dev)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy
