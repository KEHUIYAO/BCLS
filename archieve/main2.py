import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from torch.optim import lr_scheduler
import torch.utils.data as data
import os
import gzip
from PIL import Image
import random
import argparse
import numpy as np

from visualization import plot_spatio_temporal_data


class SampleDataset:
    '''
    basic deep hierarchical linear dstm
    Denote:
    Y_t is the soil moisture considering no precipitation
    S_t is the soil moisture including the precipitation
    X_t be precipitation
    Z_t be the observation soil moisture

    Y_t = f(\gamma, l, offset, Y_{t-1}) + \eta_{1t}, where \eta_{1t} \sim Gau(0, R_{1t})
    S_t = Y_t + f(\theta, X_t, X_{t-1}) + \eta_{2t}, where \eta_{2t} \sim Gau(0,R_{2t})
    Z_t = Y_t + \epsilon_t, where \epsilon_t \sim Gau(0, R_{3t})

    '''

    def __init__(self, n, T, gamma, l, offset, total, frames_input, frames_output, warm_start_ind=0):
        """

        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :param frames_input: length of the observed sequences
        :param frames_output: length of the sequences to predict
        :param warm_start_ind: length of the sequences to ignore as the purpose of warm-starting
        """

        self.Z, self.precipitation = self.prepare_data(n, T, gamma, l, offset, total)
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.warm_start_ind = warm_start_ind

    def prepare_data(self, n, T, gamma, l, offset, total):
        """
        :return: the observed Z of size total x T x 1 x n x n
        """

        location_list = []  # create a list storing each spatial location
        for i in range(n):
            for j in range(n):
                location_list.append([i, j])
        location_list = np.array(location_list)

        distance_matrix_list = []  # create a list, each element stores the pairwise distance between it and every spatial location
        for i in range(n * n):
            dist = np.array([np.sqrt(x[0] ** 2 + x[1] ** 2) for x in location_list[i] - location_list])
            distance_matrix_list.append(dist)
        distance_matrix_list = np.array(distance_matrix_list)

        weights_matrix = gamma * np.exp(-(
                                                     distance_matrix_list + offset) ** 2 / l)  # create a matrix, each row stores the weights matrix between it and every spatial location

        # normalize the weights
        # def normalize(x):
        #     return x / np.sum(x)
        #
        # weights_matrix = np.apply_along_axis(normalize, 1, weights_matrix)

        # check stability of the evolving process
        w, _ = np.linalg.eig(weights_matrix)
        max_w = np.max(w)
        # if max_w == 1 or max_w > 1:
        #     print("max eigen value is %f" % max_w)
        #     raise(ValueError("change initial parameters!"))
        # else:
        #     print("max eigen value is %f" % max_w)
        #     print("valid initial parameters!")

        # error terms with spatial correlation
        sigma_eta = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L1 = np.linalg.cholesky(sigma_eta).transpose()
        sigma_epsilon = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L2 = np.linalg.cholesky(sigma_epsilon).transpose()

        # simulate obs
        Z = np.zeros((n * n, T, total))
        Y = np.zeros((n * n, T, total))
        precipitation = np.random.randn(n * n, T, total) * 5 + 10
        for i in range(total):
            eta = np.dot(L1, np.random.randn(n * n, T))
            epsilon = np.dot(L2, np.random.randn(n * n, T))
            Y[:, 0, i] = precipitation[:, 0, i] + eta[:, 0] + 100
            Z[:, 0, i] = Y[:, 0, i] + epsilon[:, 0]

            # Y[:, 0, i] = precipitation[:, 0, i]
            # Z[:, 0, i] = Y[:, 0, i]

            for t in range(1, T):
                Y[:, t, i] = np.dot(weights_matrix, (Y[:, (t-1), i])[:, None]).ravel() + precipitation[:, t, i] + 0.1 * precipitation[:, t, i] + eta[:, t]
                Z[:, t, i] = Y[:, t, i] + epsilon[:, t]
                # Y[:, t, i] = np.dot(weights_matrix, (Y[:, (t - 1), i])[:, None]).ravel() + + precipitation[:, t, i]
                # Y[:, t, i] = precipitation[:, t, i]
                # Z[:, t, i] = Y[:, t, i]

        Z = Z.reshape((n, n, T, total))  # convert data to nxnxTxtotal
        Z = Z[..., None]
        Z = Z.transpose(3, 2, 4, 0, 1)

        precipitation = precipitation.reshape((n, n, T, total))
        precipitation = precipitation[..., None]
        precipitation = precipitation.transpose(3, 2, 4, 0, 1)


        # what can we do best, assume we only miss the spatial error terms
        eta = np.dot(L1, np.random.randn(n * n, T))
        epsilon = np.dot(L2, np.random.randn(n * n, T))
        print("based on the error term, the best mse we can achieve will be above %.4f" % np.mean((eta + epsilon) ** 2))

        return Z.astype(np.float32), precipitation.astype(np.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        Z = self.Z[idx, ...]  # idx th sample
        precipitation = self.precipitation[idx, ...]

        input_for_encoder = np.concatenate([Z[self.warm_start_ind:(self.warm_start_ind + self.frames_input), ...], precipitation[self.warm_start_ind:(self.warm_start_ind + self.frames_input), ...]], axis=1)  # observed sequences

        input_for_decoder = precipitation[(self.warm_start_ind + self.frames_input): (
                    self.warm_start_ind + self.frames_input + self.frames_output), ...]


        # input_for_decoder = []

        output = Z[(self.warm_start_ind + self.frames_input): (
                    self.warm_start_ind + self.frames_input + self.frames_output), ...]  # sequences to predict
        # additional_time_invariant_inputs = precipitation[(self.warm_start_ind + self.frames_input): (self.warm_start_ind + self.frames_input + self.frames_output), ...]

        additional_time_invariant_inputs = []

        return [idx, output, input_for_encoder, input_for_decoder, additional_time_invariant_inputs]


class SampleDatasetDataModule(pl.LightningDataModule):
    def __init__(self, n, T, gamma, l, offset, batch_size, training_data_size, validation_data_size, frames_input,
                 frames_output, warm_start_ind=0):
        super().__init__()
        self.n = n
        # self.T = frames_input + frames_output
        self.T = T
        self.gamma = gamma
        self.l = l
        self.offset = offset
        self.training_data_size = training_data_size
        self.validation_data_size = validation_data_size
        self.batch_size = batch_size
        self.total = training_data_size + validation_data_size
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.warm_start_ind = warm_start_ind

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.data = SampleDataset(self.n, self.T, self.gamma, self.l, self.offset, self.total, self.frames_input,
                                      self.frames_output, self.warm_start_ind)
            self.training_data, self.validation_data = random_split(self.data, [self.training_data_size,
                                                                                self.validation_data_size])

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)


class MovingMNIST(data.Dataset):
    def __init__(self, root, n_frames_input, n_frames_output, num_digits, image_size, digit_size, N,
                 transform=None, use_fixed_dataset=False):
        '''
        if use_fixed_dataset = True, the mnist_test_seq.npy in the root folder will be loaded
        '''
        super().__init__()
        self.use_fixed_dataset = use_fixed_dataset
        if not use_fixed_dataset:
            self.mnist = self.load_mnist(root, image_size=digit_size)
        else:
            self.dataset = self.load_fixed_set(root)

            # take a slice
            assert (self.dataset.shape[1] > N)
            self.dataset = self.dataset[:, :N, ...]

        self.length = N
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = image_size
        self.digit_size_ = digit_size
        self.step_length_ = 0.1
        self.num_digits = num_digits

    def load_mnist(self, root, image_size):
        # Load MNIST dataset for generating training data.
        path = os.path.join(root, 'train-images-idx3-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            mnist = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist = mnist.reshape(-1, image_size, image_size)
        return mnist

    def load_fixed_set(self, root):
        # Load the fixed dataset
        filename = 'mnist_test_seq.npy'
        path = os.path.join(root, filename)
        dataset = np.load(path)
        dataset = dataset[..., np.newaxis]
        return dataset

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(self.num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        # Sample number of objects
        # Generate data on the fly
        if not self.use_fixed_dataset:
            images = self.generate_moving_mnist()
        else:
            images = self.dataset[:, idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1
        w = int(self.image_size_ / r)
        # w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length


class MovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self, root='data/', batch_size=1, training_data_size=1, validation_data_size=1, frames_input=10,
                 frames_output=10, num_digits=2, image_size=64, digit_size=28, N=10,
                 transform=None, use_fixed_dataset=True):
        super().__init__()
        self.root = root
        self.training_data_size = training_data_size
        self.validation_data_size = validation_data_size
        self.batch_size = batch_size
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.image_size = image_size
        self.digit_size = digit_size
        self.N = training_data_size + validation_data_size
        self.transform = transform
        self.use_fixed_dataset = use_fixed_dataset

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data = MovingMNIST(root=self.root,
                                    n_frames_input=self.frames_input,
                                    n_frames_output=self.frames_output,
                                    image_size=self.image_size,
                                    digit_size=28,
                                    N=self.N,
                                    transform=self.transform,
                                    use_fixed_dataset=self.use_fixed_dataset
                                    )
            self.training_data, self.validation_data = random_split(self.data, [self.training_data_size,
                                                                                self.validation_data_size])

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)


class BayesianDropout(pl.LightningModule):
    """a variant of dropout which makes the model become a Bayesian neural network
     Implementation of BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE by Yarin Gal. The core idea is to set an approximating distribution modelling each kernel-patch pair with a distinct random variable, and this distribution randomly sets kernels to zero for different patches, which results in the equivalent explanation of applying dropout for each element in the tensor y before pooling. So implementing the bayesian CNN is therefore as simple as using dropout after every convolution layer before pooling
    """

    def __init__(self, dropout, x):
        "generate dropout mask using x's shape"
        super().__init__()
        self.dropout = dropout
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m = x.new_empty(x.size()).bernoulli_(1 - dropout).to(device)
        # self.m = x.new_empty(x.size()).bernoulli_(1 - dropout)

    def forward(self, x):
        "apply the dropout mask to x"
        x = x.masked_fill(self.m == 0, 0)
        return x


class CLSTM_cell(pl.LightningModule):
    """
    singler layer of ConvLSTMCell
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

    def forward(self, seq_len, inputs, hidden_state):
        """
        inputs is of size (S, B, C, H, W)
        hidden state is of size (B, C_new, H, W)
        seq_len=10 for moving_mnist
        return a turple of (a, (b, c)), where a is of shape (S, B, C_new, H, W); b and c are of shape (B, C_new, H, W)
        """

        # if hidden_state is None, initialize it with zeros
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
        else:
            hx, cx = hidden_state

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
            if inputs is None:
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


class ConvCell(pl.LightningModule):
    """
    used to apply separate CNN for images at different time steps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate):
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

        self.dropout_rate = dropout_rate

    def forward(self, x):
        "x is of size B, S, C, H, W"

        # make x to be size B*S, C, H, W
        B, S, C, H, W = x.size()
        x = torch.reshape(x, (-1, C, H, W))

        # forward through CNN
        x = self.conv2d(x)

        # apply CNN version of bayesian dropout
        if self.dropout_rate == 0:
            pass
        else:
            dropout_layer = BayesianDropout(self.dropout_rate, x)
            x = dropout_layer(x)

        # apply maxpooling
        x = self.pooling_layer(x)

        # apply non-linearity function
        # x = self.relu_layer(x)

        # make x to be size B, S, C_new, H_new, W_new
        C_new = x.size(1)
        H_new = x.size(2)
        W_new = x.size(3)
        x = torch.reshape(x, (B, S, C_new, H_new, W_new))

        return x


class Encoder(pl.LightningModule):
    """
    used to encode the data.
    consists of multiple ConvLSTM cells
    """

    def __init__(self, rnns):
        super().__init__()
        self.blocks = len(rnns)

        # rnn is a ConvLSTM cell
        for index, rnn in enumerate(rnns, 1):
            # index sign from 1
            setattr(self, 'rnn' + str(index), rnn)

    def forward(self, inputs):

        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        T = inputs.size(0)
        hidden_states = []

        for i in range(1, self.blocks + 1):
            cur_rnn = getattr(self, 'rnn' + str(i))
            inputs, state_stage = cur_rnn(seq_len=T, inputs=inputs, hidden_state=None)
            hidden_states.append(state_stage)

        return tuple(hidden_states)


class Decoder(pl.LightningModule):
    """
    used to decode data.
    consists of multiple ConvLSTM cells and one ConvCell mapping the hidden state to output
    """

    def __init__(self, rnns, cnn):
        super().__init__()
        self.blocks = len(rnns)

        for index, rnn in enumerate(rnns, 1):
            setattr(self, 'rnn' + str(index), rnn)

        # the output layer is a ConvCell
        self.output_layer = cnn




    def forward(self, hidden_states, inputs, additional_time_invariant_inputs):

        if len(inputs) == 0 and len(additional_time_invariant_inputs) == 0:
            raise(ValueError('decoder receives none'))

        if len(inputs) > 0:
            inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
            T = inputs.size(0)
        else:
            inputs = None
            T = additional_time_invariant_inputs.size(1)



        cur_rnn = getattr(self, 'rnn1')
        res = []
        inputs, _ = cur_rnn(seq_len=T, inputs=inputs, hidden_state=hidden_states[0])
        res.append(inputs)
        for i in list(range(1, self.blocks)):
            cur_rnn = getattr(self, 'rnn' + str(i + 1))
            inputs, _ = cur_rnn(seq_len=T, inputs=inputs, hidden_state=hidden_states[i])
            res.append(inputs)

        # append the channels of all the layers of the decoder network
        inputs = torch.cat(res, dim=2)

        inputs = inputs.transpose(0, 1)  # to B,S,C_sum,H,W

        # additional layer for including time-invariant inputs
        if len(additional_time_invariant_inputs) > 0:
            inputs = torch.cat([inputs, additional_time_invariant_inputs], dim=2)


        outputs = self.output_layer(inputs)



        return outputs


class ED(pl.LightningModule):
    """encoder-decoder network
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input):
        """forward pass of the ED net
        If it is the prediction task, we first encode a sequence using the encoder network, then pass the output of the encoder to the decoder network and obtain the output.
        If it is the gap-filling task, for example, the length of the original sequence is 8, and the mask is 11001111, and our goal is to gap-fill the third and the forth time step based on the observed sequence. In this example, an encoder network is used to encode the first two time steps, and pass its output to the decoder, then the decoder network will decode two steps and output both the predictions and hidden states used as the initial state for the next encoder, and the next encoder will continue to encode information until it finds a missing time step in the data and calls for the decoder.


        :param input_for_encoder: a list of tensors, each tensor is of shape (B, S, C1, H, W)
        :param input_for_decoder: a list of tensors, each tensor is of shape (B, S, C2, H, W)
        :param additional_time_invariant_input: a list of tensors, each tensor is of shape (B, S, C3, H, W)
        :return:
        """
        state = self.encoder(inputs=input_for_encoder)
        output = self.decoder(hidden_states=state, inputs=input_for_decoder,additional_time_invariant_inputs=additional_time_invariant_input)
        return output


class LightningConvLstm(pl.LightningModule):
    """
    functions includes:
    model checkpointing
    built-in gpu training
    logging
    visualization
    early stopping
    distributed training
    """

    def __init__(self, encoder_rnns, decoder_rnns, output_cnn, mc_dropout=5,
                 learning_rate=1e-3):
        super(LightningConvLstm, self).__init__()
        self.encoder = Encoder(encoder_rnns)
        self.decoder = Decoder(decoder_rnns, output_cnn)
        self.net = ED(self.encoder, self.decoder)

        self.loss_function = nn.MSELoss()
        # self.loss_function = nn.CrossEntropyLoss()
        self.mc_dropout = mc_dropout
        self.learning_rate = learning_rate

    def simple_plot(self, x, pred, text='Train/Pred'):

        " x is a tensor of size (S, C, H, W)"
        # detach
        x = torch.Tensor.cpu(x).detach()
        pred = torch.Tensor.cpu(pred).detach()

        # grid_x = torchvision.utils.make_grid(x)
        # grid_pred = torchvision.utils.make_grid(pred)

        # self.logger.experiment.add_image("True"+text, grid_x, self.current_epoch)
        # self.logger.experiment.add_image("Pred"+text, grid_pred, self.current_epoch)

        self.logger.experiment.add_figure("True" + text, plot_spatio_temporal_data(x.squeeze(1)), self.current_epoch)
        self.logger.experiment.add_figure("Pred" + text, plot_spatio_temporal_data(pred.squeeze(1)), self.current_epoch)

    def simple_plot_std(self, x, text='Std'):

        " x is a tensor of size (S, C, H, W)"
        # detach
        x = torch.Tensor.cpu(x).detach()


        # grid_x = torchvision.utils.make_grid(x)
        # grid_pred = torchvision.utils.make_grid(pred)

        # self.logger.experiment.add_image("True"+text, grid_x, self.current_epoch)
        # self.logger.experiment.add_image("Pred"+text, grid_pred, self.current_epoch)

        self.logger.experiment.add_figure("Pred" + text, plot_spatio_temporal_data(x.squeeze(1)), self.current_epoch)


    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input):
        " x is of shape (B, S, C, J, W)"
        return self.net(input_for_encoder, input_for_decoder, additional_time_invariant_input)

    def training_step(self, batch, batch_idx):
        # attrs = vars(self)
        # print(', '.join("%s: %s" % item for item in attrs.items()))
        # attrs = vars(self.trainer)
        # print(', '.join("%s: %s" % item for item in attrs.items()))
        (idx, targetVar, input_for_encoder, input_for_decoder, additional_time_invariant_input) = batch
        pred = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input)  # (B,S,C,H,W)
        # print(targetVar[0,0,0,...])
        loss = self.loss_function(pred, targetVar)
        # self.log('train_loss', loss)

        # if it's the last batch in the current epoch, record the true image sequence and the predicted image sequences
        if self.trainer.is_last_batch:
            # add one image sequence in a batch and the corresponding predicted image
            # only use the first image sequence in a batch
            pred = pred[0, ...]
            targetVar = targetVar[0, ...]
            self.simple_plot(targetVar, pred, text='Train')

        return loss

    def training_epoch_end(self, outputs):
        "the function is called after every epoch is completed"

        # add computational graph
        # if (self.current_epoch == 1):
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     sampleImg = torch.rand((1, 10, 1, 64, 64)).to(device)
        #     self.logger.experiment.add_graph(LightningConvLstm(encoder_rnns, decoder_rnns, output_cnn), sampleImg)

        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # attrs = vars(self.trainer)
        # print(', '.join("%s: %s" % item for item in attrs.items()))
        (idx, targetVar, input_for_encoder, input_for_decoder, additional_time_invariant_input) = batch
        res = []
        for i in range(self.mc_dropout):
            pred = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input)  # B,S,C,H,W
            res.append(pred)
        pred_avg = torch.stack(res).mean(dim=0)
        pred_std = torch.stack(res).std(dim=0)

        loss = self.loss_function(pred_avg, targetVar)

        # self.log('validation_loss', loss)
        naive_predictor = torch.ones_like(targetVar).permute(3, 4, 0, 1, 2) * targetVar.mean(dim=(-1, -2))
        naive_predictor = naive_predictor.permute(2, 3, 4, 0, 1)
        naive_predictor_loss = self.loss_function(naive_predictor, targetVar)

        # if it's the first batch in the current epoch, record the true image sequence and the predicted image sequences
        if batch_idx == 0:
            # add one image sequence in a batch and the corresponding predicted image
            # only use the first image sequence in a batch
            pred_avg = pred_avg[0, ...]
            pred_std = pred_std[0, ...]
            targetVar = targetVar[0, ...]
            self.simple_plot(targetVar, pred_avg, text='Validation')
            self.simple_plot_std(pred_std)

        return {'loss': loss, 'naive_predictor_loss': naive_predictor_loss}

    def validation_epoch_end(self, outputs):
        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_naive_predictor_loss = torch.stack([x['naive_predictor_loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Loss/Naive_predictor', avg_naive_predictor_loss, self.current_epoch)
        self.log('val_loss', avg_loss)  # metric to be tracked

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val_loss'}}


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root', type=str, default='data/')
    # parser.add_argument('--dropout_rate', type=float, default=0, help='dropout rate for all layers')
    # parser.add_argument('--training_data_size', type=int, default=20)
    # parser.add_argument('--validation_data_size', type=int, default=10)
    # parser.add_argument('--frames_input', type=int, default=10)
    # parser.add_argument('--frames_output', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--gpu', type=int, default=1, help='type 1 if you want to use gpu, type 0 if you want to use cpu')
    # parser.add_argument('--max_epoch', type=int, default=20)

    # debug locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../data/')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate for all layers')
    parser.add_argument('--training_data_size', type=int, default=100)
    parser.add_argument('--validation_data_size', type=int, default=50)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=15)
    parser.add_argument('--frames_input', type=int, default=5)
    parser.add_argument('--frames_output', type=int, default=5)
    parser.add_argument('--warm_start_ind', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0,
                        help='type 1 if you want to use gpu, type 0 if you want to use cpu')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--mc_dropout', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    ## case 1
    # encoder_rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=5, num_features=128, dropout_rate=args.dropout_rate),
    #                 CLSTM_cell(shape=(64, 64), input_channels=128, filter_size=5, num_features=64, dropout_rate=args.dropout_rate),
    #                 CLSTM_cell(shape=(64, 64), input_channels=64, filter_size=5, num_features=64, dropout_rate=args.dropout_rate)]
    # decoder_rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=5, num_features=128, dropout_rate=args.dropout_rate),
    #                 CLSTM_cell(shape=(64, 64), input_channels=128, filter_size=5, num_features=64, dropout_rate=args.dropout_rate),
    #                 CLSTM_cell(shape=(64, 64), input_channels=64, filter_size=5, num_features=64, dropout_rate=args.dropout_rate)]
    # output_cnn = ConvCell(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, dropout_rate=args.dropout_rate)

    ## case 2
    # encoder_rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=5, num_features=16, dropout_rate=args.dropout_rate)]
    # decoder_rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=5, num_features=16, dropout_rate=args.dropout_rate)]
    #
    # output_cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dropout_rate=args.dropout_rate)

    ## case 3
    encoder_rnns = [
        CLSTM_cell(shape=(10, 10), input_channels=2, filter_size=3, num_features=64, dropout_rate=args.dropout_rate),
        CLSTM_cell(shape=(10, 10), input_channels=64, filter_size=3, num_features=64, dropout_rate=args.dropout_rate)]
    decoder_rnns = [
        CLSTM_cell(shape=(10, 10), input_channels=1, filter_size=1, num_features=64, dropout_rate=args.dropout_rate),
        CLSTM_cell(shape=(10, 10), input_channels=64, filter_size=1, num_features=64, dropout_rate=args.dropout_rate)
        ]
    output_cnn = ConvCell(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, dropout_rate=0)

    # case 4
    # encoder_rnns = [
    #     CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=3, num_features=128, dropout_rate=args.dropout_rate)]
    # decoder_rnns = [
    #     CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=3, num_features=128, dropout_rate=args.dropout_rate)]
    #
    # output_cnn = ConvCell(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0,
    #                       dropout_rate=0)

    model = LightningConvLstm(encoder_rnns, decoder_rnns, output_cnn, args.mc_dropout, args.learning_rate)
    # attrs = vars(model)
    # print(', '.join("%s: %s" % item for item in attrs.items()))

    # dm = MovingMNISTDataModule(root=args.root, training_data_size=args.training_data_size, validation_data_size=args.validation_data_size, batch_size=args.batch_size, frames_input=args.frames_input, frames_output=args.frames_output, num_digits=2, image_size=64, digit_size=28, N=args.training_data_size + args.validation_data_size,
    #              transform=None, use_fixed_dataset=True)

    dm = SampleDatasetDataModule(n=args.n, T=args.seq_len, gamma=0.2, l=1, offset=0, batch_size=args.batch_size,
                                 training_data_size=args.training_data_size,
                                 validation_data_size=args.validation_data_size, frames_input=args.frames_input,
                                 frames_output=args.frames_output, warm_start_ind=args.warm_start_ind)

    #
    # model.load_from_checkpoint(checkpoint_path='tb_logs/Bayesian_ConvLSTM/version_131/checkpoints/epoch=47-step=2399.ckpt', encoder_rnns=encoder_rnns, decoder_rnns=decoder_rnns, output_cnn=output_cnn, frames_input=args.frames_input, frames_output=args.frames_output)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    if args.gpu == 0:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)

    # trainer.tune(model, dm)
    trainer.fit(model, dm)

    # test
    # prediction_list = []
    # for i, batch in enumerate(dm.val_dataloader()):
    #     (idx, targetVar, inputVar, _, _) = batch
    #     pred = model(inputVar)  # (B,S,C,H,W)
    #     prediction_list.append(pred)
    #
    # print(prediction_list[0])

    # to run on cpus: try python main.py --root='../../data' --training_data_size=1 --validation_data_size=1 --gpu=0
    # load tensorboard
    # %load_ext tensorboard
    # %tensorboard --logdir tb_logs/Bayesian_ConvLSTM






