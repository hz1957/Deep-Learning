import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Use torch tensor math function or else you will run into issues later      #
        # where the computational graph is broken and appropriate gradients cannot   #
        # be computed.                                                               #
        #                                                                            #
        # HINT: You may find torch.bmm() function is useful for processing a matrix  #
        # product in a batch.                                                        #
        ##############################################################################
        N, C, H, W = features.shape
        features_reshaped = features.view(N, C, H*W)
        gram_matrix = torch.bmm(features_reshaped, features_reshaped.transpose(1,2))

        if normalize:
            gram_matrix /= (H * W * C)

        return gram_matrix
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of the same length as style_layers, where style_targets[i] is
             a PyTorch Variable giving the Gram matrix the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """

        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Use torch tensor math function or else you will run into issues later      #
        # where the computational graph is broken and appropriate gradients cannot   #
        # be computed.                                                               #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################
        style_loss = 0
        # Iterate over the style layers:
        for layer_i, target, weight in zip(style_layers, style_targets, style_weights):
            # Compute the Gram Matrix of the current layer
            current = self.gram_matrix(feats[layer_i])
            
            # Compute the Style Loss
            style_loss += weight * torch.sum((current-target)**2)
        
        return style_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

