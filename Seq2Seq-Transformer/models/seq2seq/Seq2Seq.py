
import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device, attention=False):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.attention=attention  #if True attention is implemented
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.attention = attention
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden being fed into the decoder           #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################

        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, decoder_output_size)
        outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size)
        outputs = outputs.to(self.device)
        encoder_outputs, hidden_state = self.encoder(source)


        for t in range (0, out_seq_len):
            if t == 0:
                input_t = source[:, 0][:, None]
                #print(input_t)
            else:
                input_t = output_t.topk(1, dim=-1)[1]
                #print(output_t,"\n",output_t.topk(1, dim=-1))
            output_t, hidden_state = self.decoder(input_t, hidden_state, encoder_outputs=encoder_outputs, attention=self.attention)
            outputs[:, t, :] = output_t
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
