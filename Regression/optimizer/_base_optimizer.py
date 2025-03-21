

class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        """

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        for param_key, param_value in model.weights.items():
            # Only update the weights (not biases)
            if param_key.startswith('W'):
                # Update the gradient dictionary with the L2 regularization term
                model.gradients[param_key] += self.reg * param_value
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
