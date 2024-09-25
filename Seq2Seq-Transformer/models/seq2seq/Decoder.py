
import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.embedding = nn.Embedding(self.output_size, self.emb_size)

        if model_type == "RNN" :
            self.rnn = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True)
        self.linear = nn.Linear(self.decoder_hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        hidden = hidden.permute((1,2,0))  # (N, hidden_dim, 1) 
        encoder_outputs = encoder_outputs.transpose(1,2) # (N, hidden_dim, T)
        cosine_sim = torch.nn.functional.cosine_similarity(hidden, encoder_outputs, dim=1) # (N, 1, T)
        attention = torch.nn.functional.softmax(cosine_sim, dim=1).unsqueeze(1) # Shape: (N, 1, T)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       If attention is true, compute the attention probabilities and use   #
        #       them to do a weighted sum on the encoder_outputs to determine       #
        #       the hidden (and cell if LSTM) states that will be consumed by the   #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################
        embed = self.embedding(input)
        embed = self.dropout(embed)
        
        if self.model_type == "LSTM":
            hidden, cell = hidden

        if attention:
            attention_hidden = self.compute_attention(hidden, encoder_outputs) # Shape: (N, 1, T)
            hidden = torch.bmm(attention_hidden, encoder_outputs)  # Shape: (N, 1, encoder_hidden_size)
            hidden = hidden.transpose(0,1)
            #print(hidden.size())

        if self.model_type == "RNN":
            output, hidden = self.rnn(embed, hidden)

        elif self.model_type == "LSTM":
            if attention:
                attention_cell = self.compute_attention(cell, encoder_outputs)
                cell = torch.bmm(attention_cell, encoder_outputs)  # Shape: (N, 1, encoder_hidden_size)
                cell = cell.transpose(0,1)
            hidden = hidden, cell    
            output, hidden = self.rnn(embed, hidden)

        output = self.linear(output)
        output = self.logsoftmax(output)
        output = output.squeeze(1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
