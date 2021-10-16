"""implementing bayesian dropout layer, encoding and decoding network with convlstm structure
In the comments below, B is short for BATCH_SIZE; S is short for sequence length or temporal dimension T; C is short for channel_dimension; H is short for height; W is short for width.
"""
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
import torchvision
import torch.utils.data as data
import os
import gzip
import random
import argparse

from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import lr_scheduler
from PIL import Image

# import from my code
from visualization import plot_spatio_temporal_data

class BayesianDropout(pl.LightningModule):
    """a variant of dropout layer
     Implementation follows http://arxiv.org/abs/1506.02158 and https://arxiv.org/abs/1512.05287
    """

    def __init__(self, dropout, x):
        """generate dropout mask using x's shape
        when this layer is initialized, it creates a masking based on x's shape, if we use the same layer later, it ensures that we are
        using the same masking.
        :param dropout: dropout rate
        :param x: the template tensor
        """
        super().__init__()
        self.dropout = dropout
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m = x.new_empty(x.size()).bernoulli_(1 - dropout).to(device)


    def forward(self, x):
        "apply the dropout mask to x, notice that if x and self.m are not the same shape, self.m will be broadcast to have the same shape as x"
        x = x.masked_fill(self.m == 0, 0)
        return x

class CLSTM_cell(pl.LightningModule):
    """singler layer of ConvLSTMCell
    The convlstm structure follows https://arxiv.org/abs/1506.04214, we add dropout mechanism following http://arxiv.org/abs/1506.02158

    Attributes:
        shape: the size of the grid
        input_channels: number of input channels
        filter_size: controls the size of the convolution operator
        num_features: controls how many filters we are using
        padding: number of zero-padding around the corner
        conv: conv2d layer
        dropout_rate: dropout_rate

    """

    def __init__(self, shape, input_channels, filter_size, num_features, dropout_rate):
        super(CLSTM_cell, self).__init__()

        # (H, W)
        self.shape = shape

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features

        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2

        # input_dim+hidden_dim -> 4*hidden_dim
        self.conv = nn.Conv2d(self.input_channels + self.num_features,
                              4 * self.num_features, self.filter_size, 1,
                              self.padding)

        # apply GroupNorm, the input channels are separated into num_groups groups, each containing num_channels / num_groups channels. The mean and standard-deviation are calculated separately over the each group.
        if 4 * self.num_features < 32:
            print("GroupNorm will not be applied, require more output channels to apply GroupNorm!")
        else:
            print("GroupNorm will be applied!")
            self.groupnorm = nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)

        self.dropout_rate = dropout_rate

    def forward(self, seq_len, inputs, initial_state):
        """forward pass of one layer convlstm
        inputs and initial_state cannot be [] simultaneously

        :param seq_len: when inputs are not [], the seq_len should be equal to inputs.size(0). Otherwise, if inputs are [], the seq_len determines how long the sequence is going to be generated
        :param inputs: Either a tensor of size (S, B, input_channels, H, W) or []
        :param initial_state: Either a list of two tensors, both are of size (B, num_features, H, W) to ensure the channel of the initial_state tensor match the output channel, Or []

        :return: a tuple of (hidden_state_list, (last_hidden_state, last_cell_state)), where hidden_state_list is of shape (S, B, num_features, H, W), which is the hidden state at each time step concatenated together, and last_hidden_state and last_cell_state are of size (B, num_features, H, W).
        """
        # if both initial_state and inputs are [], raise an error
        if len(initial_state) == 0 and len(inputs) == 0:
            raise(ValueError('Both initial_state and inputs are []'))


        # if initial_state is None, initialize it with zeros
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if len(initial_state) == 0:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
        else:
            hx, cx = initial_state

        output_inner = []

        # apply dropout (combining CNN version and RNN version of bayesian dropout)
        if self.dropout_rate == 0:
            pass
        else:
            self.dropout_layer = BayesianDropout(self.dropout_rate,
                                                 torch.zeros(hx.size(0), self.num_features * 4, self.shape[0],
                                                             self.shape[1]))

        # for each time step, perform a CNN on a slice of the sequence of images and record the hidden state and cell state
        for index in range(seq_len):
            if len(inputs) == 0:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).to(device)
            else:
                x = inputs[index, ...]

            # combining input and last hidden state
      
            combined = torch.cat((x, hx), 1)

            # apply CNN forward pass
            gates = self.conv(combined)  # gates: (B, num_features*4, H, W)

            # apply group norm
            gates = self.groupnorm(gates)

            # apply the same dropout mask at each time step
            if self.dropout_rate == 0:
                pass
            else:
                gates = self.dropout_layer(gates)

            # it should return 4 tensors: i,f,g,o following the literature of LSTM
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy

        return torch.stack(output_inner), (hy, cy)


class ConvRelu(pl.LightningModule):
    """a cnn layer + a relu layer
    apply a CNN on the sequences of frames. The same CNN is applied for each frame in the sequence
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        """x is a tensor of size (S, B, C, H, W), return a tensor of size (S, B, C_out, H_out, W_out)"""
        S, B, C, H, W = x.size()
        x = torch.reshape(x, (-1, C, H, W))
        x = self.conv2d(x)
        x = torch.reshape(x, (S, B, x.size(1), x.size(2), x.size(3)))

        # apply dropout (combining CNN version and RNN version of bayesian dropout)
        if self.dropout_rate == 0:
            pass
        else:
            self.dropout_layer = BayesianDropout(self.dropout_rate,
                                                 torch.zeros(B, x.size(2), x.size(3), x.size(4)))

        # use the same dropout for each time step thanks to the broadcastable implementation of bayesian dropout layer
            x = self.dropout_layer(x)

        x = self.relu_layer(x)

        return x


class DeconvRelu(pl.LightningModule):
    """a deconvolutional nn layer + a relu layer
      apply a deconvolutional nn on the sequences of frames. The same deconvolutional nn is applied for each frame in the sequence
     """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate):
        super().__init__()
        self.transposeConv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_layer = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """x is a tensor of size (S, B, C, H, W), return a tensor of size (S, B, C_out, H_out, W_out)"""
        S, B, C, H, W = x.size()
        x = torch.reshape(x, (-1, C, H, W))
        x = self.transposeConv2d(x)

        x = torch.reshape(x, (S, B, x.size(1), x.size(2), x.size(3)))

        # apply dropout (combining CNN version and RNN version of bayesian dropout)
        if self.dropout_rate == 0:
            pass
        else:
            self.dropout_layer = BayesianDropout(self.dropout_rate,
                                                 torch.zeros(B, x.size(2), x.size(3), x.size(4)))
            # use the same dropout for each time step thanks to the broadcastable implementation of bayesian dropout layer
            x = self.dropout_layer(x)

        x = self.relu_layer(x)

        return x



class ConvCell(pl.LightningModule):
    """The layer from hidden state to output
    apply CNN on the output of multi-layer convlstm to generate the final output
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.pooling_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        #self.relu_layer = nn.ReLU()

        # H_out = (H_in + 2 * padding - (kernel_size - 1) - 1) / stride
        # W_out can be calculated in the same way
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                                )


    def forward(self, x):
        "input tensor is of size (B, S, in_channels, H, W), return a tensor of (B, S, out_channels, H, W)"
        # make x to be size B*S, C, H, W
        B, S, C, H, W = x.size()
        x = torch.reshape(x, (-1, C, H, W))

        # forward through CNN
        x = self.conv2d(x)

        # apply maxpooling
        x = self.pooling_layer(x)

        # apply non-linearity function
        # x = self.relu_layer(x)

        # make x back to be original shape
        C_new = x.size(1)
        H_new = x.size(2)
        W_new = x.size(3)
        x = torch.reshape(x, (B, S, C_new, H_new, W_new))

        return x


class Encoder(pl.LightningModule):
    """encoding a certain length of sequence

    The Encoder network consists of multiple convlstm cells.
    """

    def __init__(self, rnns):
        "rnns are a list of convlstm cells"
        super().__init__()
        self.blocks = len(rnns)

        # rnn is a ConvLSTM cell
        for index, rnn in enumerate(rnns, 1):
            # index sign from 1
            setattr(self, 'rnn' + str(index), rnn)

    def forward(self, inputs, initial_state):
        """forward pass of the encoder

        :param inputs: a tensor of shape (B, S, C, H, W)
        :param initial_state:  Either a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell or []
        :return: a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell.
        """

        inputs = inputs.transpose(0, 1)  # to S, B, C, H, W
        T = inputs.size(0)
        hidden_states = []

        if len(initial_state) == 0:
            initial_state = [[] for i in range(self.blocks)]

        for i in range(1, self.blocks + 1):
            cur_rnn = getattr(self, 'rnn' + str(i))
            inputs, state_stage = cur_rnn(seq_len=T, inputs=inputs, initial_state=initial_state[i-1])
            hidden_states.append(state_stage)

        return tuple(hidden_states)

class Encoder_pro(pl.LightningModule):
    """encoding a certain length of sequence

    The Encoder network consists of multiple pairs of (ConvRelu, CLSTM) cells. The input will first go through a convrelu cell, and then to a convlstm cell, and then to another convrelu cell, so on and so forth.
    """

    def __init__(self, rnns, convrelus):
        "rnns are a list of convlstm cells, convrelus are a list of convrelu cells"
        super().__init__()
        assert len(rnns) == len(convrelus)
        self.blocks = len(rnns)

        # rnn is a ConvLSTM cell
        for index, (rnn, convrelu) in enumerate(zip(rnns, convrelus), 1):
            # index sign from 1
            setattr(self, 'rnn' + str(index), rnn)
            setattr(self, 'convrelu' + str(index), convrelu)

    def forward(self, inputs, initial_state):
        """forward pass of the encoder

        :param inputs: a tensor of shape (B, S, C, H, W)
        :param initial_state:  Either a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell or []
        :return: a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell.
        """


        T = inputs.size(1)
        hidden_states = []

        if len(initial_state) == 0:
            initial_state = [[] for i in range(self.blocks)]

        inputs = inputs.transpose(0, 1)  # make sure the input is S, B, C_new, H_new, W_new

        for i in range(1, self.blocks + 1):
            cur_convrelu = getattr(self, 'convrelu' + str(i))
            inputs = cur_convrelu(inputs)

            cur_rnn = getattr(self, 'rnn' + str(i))

            inputs, state_stage = cur_rnn(seq_len=T, inputs=inputs, initial_state=initial_state[i-1])
            hidden_states.append(state_stage)

        return tuple(hidden_states)


class Decoder(pl.LightningModule):
    """decode a sequence given an initial tuple of hidden states and cell states

    It consists of multiple convlstm cells and one convcell layer.


    """

    def __init__(self, rnns, cnn):
        "rnns are a list of convlstm cells, and cnn is a convcell"
        super().__init__()
        self.blocks = len(rnns)

        for index, rnn in enumerate(rnns, 1):
            setattr(self, 'rnn' + str(index), rnn)

        # the output layer is a ConvCell
        self.output_layer = cnn




    def forward(self, initial_state, seq_len, inputs, additional_time_invariant_inputs):
        """forward pass of the decoder

        :param seq_len: how long the sequence is decoded to be
        :param initial_state: a list of tuples [(h, c), ..., (h, c)]
        :param inputs: a tensor of size (B, S, C, H, W) or []
        :param additional_time_invariant_inputs: a tensor of size (B, S, C, H, W) or []
        :return: a tuple of (output, hidden_states), where the output is a tensor of size (B, S, output_channel, H, W), and the hidden_states is a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell.
        """


        if len(inputs) > 0:
            inputs = inputs.transpose(0, 1)  # to S, B, C, H, W

        cur_rnn = getattr(self, 'rnn1')
        res = []
        hidden_states = []
        inputs, state_stage = cur_rnn(seq_len=seq_len, inputs=inputs, initial_state=initial_state[0])
        res.append(inputs)
        hidden_states.append(state_stage)

        for i in list(range(1, self.blocks)):
            cur_rnn = getattr(self, 'rnn' + str(i + 1))
            inputs, state_stage = cur_rnn(seq_len=seq_len, inputs=inputs, initial_state=initial_state[i])
            res.append(inputs)
            hidden_states.append(state_stage)

        # append the channels of all the convlstm cells
        inputs = torch.cat(res, dim=2)

        inputs = inputs.transpose(0, 1)  # to B,S,C_sum,H,W

        # additional layer for including time-invariant inputs
        if len(additional_time_invariant_inputs) > 0:
            inputs = torch.cat([inputs, additional_time_invariant_inputs], dim=2)


        outputs = self.output_layer(inputs)



        return (outputs, hidden_states)


class Decoder_pro(pl.LightningModule):
    """decode a sequence given an initial tuple of hidden states and cell states

    It consists of multiple (convlstm, deconvrelu) pairs and one convcell. The inputs will first pass through an convlstm cell, then pass through a deconvrelu cell, and then to another convlstm cell, so on so forth. Finally, the inputs will go through a convcell and get the output.
    """

    def __init__(self, rnns, deconvrelus, cnn):
        "rnns are a list of convlstm cells, deconvrelus are a list of deconvrelu cells and cnn is a convcell"
        super().__init__()
        self.blocks = len(rnns)

        for index, (rnn, deconvrelu) in enumerate(zip(rnns, deconvrelus), 1):
            setattr(self, 'rnn' + str(index), rnn)
            setattr(self, 'deconvrelu' + str(index), deconvrelu)

        # the output layer is a ConvCell
        self.output_layer = cnn




    def forward(self, initial_state, seq_len, inputs, additional_time_invariant_inputs):
        """forward pass of the decoder_pro

        :param seq_len: how long the sequence is decoded to be
        :param initial_state: a list of tuples [(h, c), ..., (h, c)]
        :param inputs: a tensor of size (B, S, C, H, W) or []
        :param additional_time_invariant_inputs: a tensor of size (B, S, C, H, W) or []
        :return: a tuple of (output, hidden_states), where the output is a tensor of size (B, S, output_channel, H, W), and the hidden_states is a list containing [(h, c), ..., (h, c)], where each (h, c) is the final output of a convlstm cell.
        """


        if len(inputs) > 0:
            inputs = inputs.transpose(0, 1)  # to S, B, C, H, W
        cur_deconvrelu = getattr(self, 'deconvrelu1')
        cur_rnn = getattr(self, 'rnn1')
        res = []
        hidden_states = []

        inputs, state_stage = cur_rnn(seq_len=seq_len, inputs=inputs, initial_state=initial_state[0])
        res.append(inputs)
        hidden_states.append(state_stage)

        inputs = cur_deconvrelu(inputs)


        for i in list(range(1, self.blocks)):
            cur_rnn = getattr(self, 'rnn' + str(i + 1))
            cur_deconvrelu = getattr(self, 'deconvrelu' + str(i + 1))
            inputs, state_stage = cur_rnn(seq_len=seq_len, inputs=inputs, initial_state=initial_state[i])
            res.append(inputs)
            hidden_states.append(state_stage)
            inputs = cur_deconvrelu(inputs)

        # append the channels of all the convlstm cells
        # inputs = torch.cat(res, dim=2)

        inputs = inputs.transpose(0, 1)  # to B,S,C_sum,H,W

        # additional layer for including time-invariant inputs
        if len(additional_time_invariant_inputs) > 0:
            inputs = torch.cat([inputs, additional_time_invariant_inputs], dim=2)


        outputs = self.output_layer(inputs)



        return (outputs, hidden_states)



class ED(pl.LightningModule):
    """encoder-decoder network
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len):
        """forward pass of the ED net
        If it is the prediction task, we first encode a sequence using the encoder network, then pass the output of the encoder to the decoder network and obtain the output.
        If it is the gap-filling task, for example, the length of the original sequence is 8, and the mask is 11001111, and our goal is to gap-fill the third and the forth time step based on the observed sequence. In this example, an encoder network is used to encode the first two time steps, and pass its output to the decoder, then the decoder network will decode two steps and output both the predictions and hidden states used as the initial state for the next encoder (which is the same encoder as the first one), and the next encoder will continue to encode information until it finds a missing time step in the data and calls for the decoder. In this case, we will have multiple inputs for the encoder and decoder.


        :param input_for_encoder: a list of tensors, each tensor is of shape (B, S, C1, H, W)
        :param input_for_decoder: a list of tensors, each tensor is of shape (B, S, C2, H, W) or []
        :param additional_time_invariant_input: a list of tensors, each tensor is of shape (B, S, C3, H, W) or []
        :param seq_len: a list of int, each int tells how long we should decode the sequence to be
        :return: a list of tensors, each tensor is of shape (B, S, C4, H, W), which is the prediction of the missing sequence
        """
        if len(input_for_decoder) == 0:
            input_for_decoder = [[] for i in range(len(seq_len))]

        if len(additional_time_invariant_input) == 0:
            additional_time_invariant_input = [[] for i in range(len(seq_len))]

        if len(input_for_encoder) != len(input_for_decoder):
            input_for_encoder = input_for_encoder[:-1]


        # start with an encoder
        initial_state = []
        output_list = []
        for i in range(len(input_for_encoder)):
            initial_state = self.encoder(input_for_encoder[i], initial_state=initial_state)
            output, initial_state = self.decoder(initial_state, seq_len[i], input_for_decoder[i], additional_time_invariant_input[i])
            output_list.append(output)

        return output_list

class ED_pro(pl.LightningModule):
    """encoder_pro-decoder_pro network with
    """
    def __init__(self, encoder_pro, decoder_pro):
        super().__init__()
        self.encoder_pro = encoder_pro
        self.decoder_pro = decoder_pro

    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len):
        """forward pass of the ED net
        If it is the prediction task, we first encode a sequence using the encoder network, then pass the output of the encoder to the decoder network and obtain the output.
        If it is the gap-filling task, for example, the length of the original sequence is 8, and the mask is 11001111, and our goal is to gap-fill the third and the forth time step based on the observed sequence. In this example, an encoder network is used to encode the first two time steps, and pass its output to the decoder, then the decoder network will decode two steps and output both the predictions and hidden states used as the initial state for the next encoder (which is the same encoder as the first one), and the next encoder will continue to encode information until it finds a missing time step in the data and calls for the decoder. In this case, we will have multiple inputs for the encoder and decoder.


        :param input_for_encoder: a list of tensors, each tensor is of shape (B, S, C1, H, W)
        :param input_for_decoder: a list of tensors, each tensor is of shape (B, S, C2, H, W) or []
        :param additional_time_invariant_input: a list of tensors, each tensor is of shape (B, S, C3, H, W) or []
        :param seq_len: a list of int, each int tells how long we should decode the sequence to be
        :return: a list of tensors, each tensor is of shape (B, S, C4, H, W), which is the prediction of the missing sequence
        """
        if len(input_for_decoder) == 0:
            input_for_decoder = [[] for i in range(len(seq_len))]

        if len(additional_time_invariant_input) == 0:
            additional_time_invariant_input = [[] for i in range(len(seq_len))]

        if len(input_for_encoder) != len(input_for_decoder):
            input_for_encoder = input_for_encoder[:-1]


        # start with an encoder
        initial_state = []
        output_list = []
        for i in range(len(input_for_encoder)):
            initial_state = self.encoder_pro(input_for_encoder[i], initial_state=initial_state)

            initial_state = initial_state[::-1]  # the hidden state and cell state should be reversed to send to the decoder pro network
            output, initial_state = self.decoder_pro(initial_state, seq_len[i], input_for_decoder[i], additional_time_invariant_input[i])
            initial_state = initial_state[::-1]  # the hidden state and cell state should be reversed again to send to the encoder pro network

            output_list.append(output)

        return output_list



class Bidirectional_ED_pro(pl.LightningModule):
    """This class is particularly targeted for missing frames imputation
    In a forward pass, An ED_pro is used to predict the missing frames; In a backward pass, another ED_pro is used to predict the missing frames.
    """
    def __init__(self, ED_pro_forward, ED_pro_backward):
        super().__init__()
        self.ED_pro_forward = ED_pro_forward
        self.ED_pro_backward = ED_pro_backward


    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len):
        """forward pass of the Bidirectional_ED_pro net
         It's particularly designed to impute the missing frame. We should have both inputs in the beginning and in the end, and predict the output in the middle twice, one by forward pass ED_pro, another one by backward pass ED_pro.
        :param input_for_encoder: a list of tensors, each tensor is of shape (B, S, C1, H, W)
        :param input_for_decoder: a list of tensors, each tensor is of shape (B, S, C2, H, W) or []
        :param additional_time_invariant_input: a list of tensors, each tensor is of shape (B, S, C3, H, W) or []
        :param seq_len: a list of int, each int tells how long we should decode the sequence to be
        :return: two list of tensors, each tensor is of shape (B, S, C4, H, W), which is the prediction of the missing sequence from the forward pass and from the backward pass
        """
        forward_prediction = self.ED_pro_forward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)

        # reverse the input_for_encoder
        input_for_encoder = input_for_encoder[::-1]
        input_for_encoder = [torch.flip(x,[1]) for x in input_for_encoder]
        input_for_decoder = input_for_decoder[::-1]
        input_for_decoder = [torch.flip(x,[1]) for x in input_for_decoder]
        additional_time_invariant_input = additional_time_invariant_input[::-1]
        additional_time_invariant_input = [torch.flip(x,[1]) for x in additional_time_invariant_input]
        seq_len = seq_len[::-1]

        backward_prediction = self.ED_pro_backward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)

        # make the backward pass prediction in the forward order
        backward_prediction = backward_prediction[::-1]
        backward_prediction = [torch.flip(x, [1]) for x in backward_prediction]

        return (forward_prediction, backward_prediction)







# test BayesianDropout
def test_bayesian_dropout():
    T = 2
    S = 10
    x = torch.randn(T, S)
    one_time_step = x[0, :]
    layer = BayesianDropout(0.5, one_time_step)
    one_time_step_after_dropout = layer(one_time_step)
    second_time_step = x[1, :]
    second_time_step_after_dropout = layer(second_time_step)

# test CLSTM cell
def test_CLSTM_cell():
    S = 10
    B = 2
    input_channels = 2
    H = 5
    W = 5
    CLSTM_layer = CLSTM_cell(shape=(H, W), input_channels=input_channels, filter_size=1, num_features=16, dropout_rate=0.5)
    x = torch.randn(S, B, input_channels, H, W)
    output = CLSTM_layer(S, x, [])  # forward pass


def test_ConvRelu():
    x = torch.randn(2, 10, 3, 64, 64)
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 2
    padding = 1
    dropout_rate = 0.1
    conv_relu_layer = ConvRelu(in_channels, out_channels, kernel_size, stride, padding, dropout_rate)
    y = conv_relu_layer(x)  # y should be (2, 10, 64, 32, 32)


def test_DeconvRelu():
    x = torch.randn(2, 10, 3, 16, 16)
    in_channels = 3
    out_channels = 64
    kernel_size = 4
    stride = 2
    padding = 1
    dropout_rate = 0.1
    deconv_relu_layer = DeconvRelu(in_channels, out_channels, kernel_size, stride, padding, dropout_rate)
    y = deconv_relu_layer(x)  # y should be 2, 10, 64, 32, 32



def test_ConvCell():
    # test whether reshape function works as we want
    x = torch.randn(2, 2, 2)
    y = torch.reshape(x, (-1, 2))
    z = torch.reshape(y, (2, 2, 2))
    assert np.array_equal(x, z)
    x = torch.randn(2, 10, 3, 5, 5)
    ConvCell_layer = ConvCell(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
    y = ConvCell_layer(x)

# test Encoder
def test_encoder():
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=2, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]  # the first convlstm cell's out_channel should be equal to the second convlstm cell's input_channel
    encoder_net = Encoder(rnns)
    S = 10
    B = 2
    input_channels = 2
    H = 5
    W = 5
    x = torch.randn(B, S, input_channels, H, W)
    y1 = encoder_net(x, initial_state=[])
    y2 = encoder_net(x, initial_state=y1)

# test Decoder
def test_decoder():
    seq_len = 10
    B = 2
    input_channels = 2
    H = 5
    W = 5
    rnns = [CLSTM_cell(shape=(H, W), input_channels=input_channels, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(H, W), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]  # the first convlstm cell's out_channel should be equal to the second convlstm cell's input_channel
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)  # the in_channels for the convcell should be equal to the sum of out_channels among all convlstm cells in rnns plus the additional time-invariant input dimension
    decoder_net = Decoder(rnns, cnn)

    h1 = torch.randn(B, 16, H, W)  # the initial hidden and cell state's channel should match the output channels of the convlstm cell
    c1 = torch.randn(B, 16, H, W)
    h2 = torch.randn(B, 16, H, W)
    c2 = torch.randn(B, 16, H, W)
    initial_state = [(h1, c1), (h2, c2)]
    y = decoder_net(initial_state=initial_state, seq_len=seq_len, inputs=[], additional_time_invariant_inputs=[])

    # test Decoder with inputs
    x = torch.randn(B, seq_len, 2, H, W)
    y = decoder_net(initial_state=initial_state, seq_len=seq_len, inputs=x, additional_time_invariant_inputs=[])
    output = y[0]
    hidden_states = y[1]

    # test Decoder with additional time invariant inputs
    x1 = torch.randn(B, seq_len, 2, H, W)
    x2 = torch.randn(B, seq_len, 1, H, W)
    cnn = ConvCell(in_channels=33, out_channels=1, kernel_size=1, stride=1, padding=0)  # the in_channels for the convcell should be equal to the sum of out_channels among all convlstm cells in rnns plus the additional time-invariant input dimension
    decoder_net = Decoder(rnns, cnn)
    y = decoder_net(initial_state=initial_state, seq_len=seq_len, inputs=x1, additional_time_invariant_inputs=x2)

def test_encoder_pro():
    rnns = [CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64, dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5) ]

    convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.1),
                 ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.1),
                 ConvRelu(96, 96, 3, 2, 1, dropout_rate=0.1)]

    encoder_net = Encoder_pro(rnns, convrelus)
    S = 10
    B = 2
    input_channels = 1
    H = 64
    W = 64
    x = torch.randn(B, S, input_channels, H, W)
    y1 = encoder_net(x, initial_state=[])
    y2 = encoder_net(x, initial_state=y1)


def test_decoder_pro():
    seq_len = 3
    B = 2
    rnns = [CLSTM_cell(shape=(16, 16), input_channels=1, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64, dropout_rate=0.5)]

    deconvrelus = [DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.1)]

    cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

    h1 = torch.randn(B, 64, 64, 64)
    c1 = torch.randn(B, 64, 64, 64)
    h2 = torch.randn(B, 96, 32, 32)
    c2 = torch.randn(B, 96, 32, 32)
    h3 = torch.randn(B, 96, 16, 16)
    c3 = torch.randn(B, 96, 16, 16)
    initial_state = [(h3, c3), (h2, c2), (h1, c1)]  # reverse
    y = decoder_net(initial_state=initial_state, seq_len=seq_len, inputs=[], additional_time_invariant_inputs=[])

    # test Decoder with input
    x = torch.randn(B, seq_len, 1, 16, 16)
    y = decoder_net(initial_state=initial_state, seq_len=seq_len, inputs=x, additional_time_invariant_inputs=[])






# test Encoder-Decoder network, gap-filling task
def test_ED():
    B = 2
    C1 = 3
    C2 = 2
    H = 5
    W = 5
    x1 = torch.randn(B, 1, C1, H, W)
    x2 = torch.randn(B, 2, C1, H, W)
    xx1 = torch.randn(B, 3, C2, H, W)
    xx2 = torch.randn(B, 4, C2, H, W)
    input_for_encoder = [x1, x2]
    input_for_decoder = [xx1, xx2]
    seq_len = [3, 4]
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=3, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=2, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)
    output_list = ED_net(input_for_encoder=input_for_encoder, input_for_decoder=input_for_decoder, additional_time_invariant_input=[], seq_len=seq_len)
    y1 = output_list[0]
    y2 = output_list[1]


    # test Encoder-Decoder network, prediction task
    x = torch.randn(B, 5, C1, H, W)
    xx = torch.randn(B, 10, C2, H, W)
    input_for_encoder = [x]
    input_for_decoder = [xx]
    seq_len = [5, 10]
    output_list = ED_net(input_for_encoder=input_for_encoder, input_for_decoder=input_for_decoder, additional_time_invariant_input=[], seq_len=seq_len)
    y = output_list[0]

# test Encoder_pro and Decoder_pro network
def test_ED_pro():
    # encoder pro
    rnns = [CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64,
                       dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5) ]

    convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.1),
                 ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.1),
                 ConvRelu(96, 96, 3, 2, 1, dropout_rate=0.1)]

    encoder_net = Encoder_pro(rnns, convrelus)

    # input for encoder
    S = 10
    B = 2
    input_channels = 1
    H = 64
    W = 64
    input_for_encoder = [torch.randn(B, S, input_channels, H, W)]

    # decoder_pro
    rnns = [CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64, dropout_rate=0.5)]

    deconvrelus = [DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.1)]

    cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

    # ED net
    ED_net = ED_pro(encoder_net, decoder_net)
    seq_len = [10]
    output_list = ED_net(input_for_encoder=input_for_encoder, input_for_decoder=[], additional_time_invariant_input=[], seq_len=seq_len)


def test_Bidirectional_ED_pro():
    # encoder pro
    rnns = [CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64,
                       dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5) ]

    convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.1),
                 ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.1),
                 ConvRelu(96, 96, 3, 2, 1, dropout_rate=0.1)]

    encoder_net = Encoder_pro(rnns, convrelus)

    # input for encoder
    S = 10
    B = 2
    input_channels = 1
    H = 64
    W = 64
    input_for_encoder = [torch.randn(B, S, input_channels, H, W), torch.randn(B, S, input_channels, H, W)]

    # decoder_pro
    rnns = [CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64, dropout_rate=0.5)]

    deconvrelus = [DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.1)]

    cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

    # ED net
    ED_net_forward = ED_pro(encoder_net, decoder_net)
    ED_net_backward = ED_pro(encoder_net, decoder_net)


    # bidirectional ED_pro
    model = Bidirectional_ED_pro(ED_net_forward, ED_net_backward)
    seq_len = [10]
    forward_pass_output, backward_pass_output = model(input_for_encoder=input_for_encoder, input_for_decoder=[], additional_time_invariant_input=[], seq_len=seq_len)



if __name__ == "__main__":
    # test_bayesian_dropout()
    # test_CLSTM_cell()
    # test_ConvCell()
    # test_encoder()
    # test_decoder()
    # test_ED()
    # test_ED_pro()
    # encoder pro
    rnns = [CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64,
                       dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5) ]

    convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.1),
                 ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.1),
                 ConvRelu(96, 96, 3, 2, 1, dropout_rate=0.1)]

    encoder_net = Encoder_pro(rnns, convrelus)

    # input for encoder
    S = 10
    B = 2
    input_channels = 1
    H = 64
    W = 64
    input_for_encoder = [torch.randn(B, S, input_channels, H, W), torch.randn(B, S, input_channels, H, W)]

    # decoder_pro
    rnns = [CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.5),
            CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64, dropout_rate=0.5)]

    deconvrelus = [DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.1),
                   DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.1)]

    cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

    # ED net
    ED_net_forward = ED_pro(encoder_net, decoder_net)
    ED_net_backward = ED_pro(encoder_net, decoder_net)


    # bidirectional ED_pro
    model = Bidirectional_ED_pro(ED_net_forward, ED_net_backward)
    seq_len = [10]
    forward_pass_output, backward_pass_output = model(input_for_encoder=input_for_encoder, input_for_decoder=[], additional_time_invariant_input=[], seq_len=seq_len)
