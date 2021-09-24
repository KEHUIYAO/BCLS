"""simulate dataset from different kinds of models
"""
import numpy as np
import os
import gzip
import random
from visualization import plot_spatio_temporal_data


class DatasetDstm:
    """class for generating data from a basic dynamical spatio-temporal model

    Denote:
    Y_t: a vector containing the underlying process of all spatial locations at time t
    X_t: a vector containing the precipitation at all spatial locations at time t
    S_t = X_t + Y_t
    Z_t = S_t + sptial_error_term

    Y_t(s) = \Sum_{x=1}^{n^2} m(s, x|\gamma, offset, l) * Y_{t-1}(x) + \eta_{t}, where \eta_{t} \sim Gau(0, R_{1t})
    and m(s, x | \gamma, offset, l) = \gamma * \exp(-(x - s + offset)^2 / l)
    S_t = Y_t + X_t
    Z_t = S_t + \epsilon_t, where \epsilon_t \sim Gau(0, R_{2t})


    Attributes:
          Z: generated dataset
          mask: a numpy vector of T containing 0 and 1s, indicating the missing patterns of the sequence
          baseline_underlying: Y_0(s)
          baseline_precipitation: average precipitation through time

    """

    def __init__(self,
                 n,
                 T,
                 gamma,
                 l,
                 offset,
                 total,
                 mask,
                 baseline_underlying,
                 baseline_precipitation
                 ):
        """
        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :param mask: a numpy vector of T containing 0 and 1s, indicating the missing patterns of the sequence
        :param baseline_underlying: Y_0(s)
        :param baseline_precipitation: average precipitation through time

        """


        self.mask = mask
        self.baseline_underlying = baseline_underlying
        self.baseline_precipitation = baseline_precipitation

        self.Z, self.X = self.prepare_data(n, T, gamma, l, offset, total)




    def prepare_data(self, n, T, gamma, l, offset, total):
        """generate data from the dstm model


        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :return: a tuple of two tensors, both of size (total x T x 1 x n x n)
        """



        location_list = []  # create a list storing each spatial location
        for i in range(n):
            for j in range(n):
                location_list.append([i, j])
        location_list = np.array(location_list)

        distance_matrix_list = [
        ]  # create a list, each element stores the pairwise distance between it and every spatial location
        for i in range(n * n):
            dist = np.array([
                np.sqrt(x[0]**2 + x[1]**2)
                for x in location_list[i] - location_list
            ])
            distance_matrix_list.append(dist)
        distance_matrix_list = np.array(distance_matrix_list)

        weights_matrix = gamma * np.exp(
            -(distance_matrix_list + offset)**2 / l
        )  # create a matrix, each row stores the weights matrix between it and every spatial location

        ## normalize the weights
        # def normalize(x):
        #     return x / np.sum(x)
        #
        # weights_matrix = np.apply_along_axis(normalize, 1, weights_matrix)

        # check stability of the evolving process
        w, _ = np.linalg.eig(weights_matrix)
        max_w = np.max(w)
        if max_w == 1 or max_w > 1:
            print("max eigen value is %f" % max_w)
            raise(ValueError("change initial parameters!"))
        else:
            print("max eigen value is %f" % max_w)
            print("valid initial parameters!")

        # error terms with spatial correlation
        sigma_eta = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L1 = np.linalg.cholesky(sigma_eta).transpose()
        sigma_epsilon = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L2 = np.linalg.cholesky(sigma_epsilon).transpose()

        # simulate obs
        Z = np.zeros((n * n, T, total))
        Y = np.zeros((n * n, T, total))
        S = np.zeros((n * n, T, total))
        precipitation = np.random.randn(n * n, T, total) + self.baseline_precipitation


        for i in range(total):
            eta = np.dot(L1, np.random.randn(n * n, T))
            epsilon = np.dot(L2, np.random.randn(n * n, T))
            Y[:, 0, i] = eta[:, 0] + self.baseline_underlying
            S[:, 0, i] = Y[:, 0, i] + precipitation[:, 0, i]
            Z[:, 0, i] = S[:, 0, i] + epsilon[:, 0]

            for t in range(1, T):
                Y[:, t, i] = np.dot(weights_matrix, (Y[:, (t - 1),
                                                     i])[:, None]).ravel() + eta[:, t]
                S[:, t, i] = Y[:, t, i] + precipitation[:, t, i]
                Z[:, t, i] = S[:, t, i ] + epsilon[:, t]

        Z = Z.reshape((n, n, T, total))  # convert data to n x n x T x total
        Z = Z[..., None]
        Z = Z.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n

        precipitation = precipitation.reshape((n, n, T, total))  # convert data to n x n x T x total
        precipitation = precipitation[..., None]
        precipitation = precipitation.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n

        # the best we can do is that we fit everything except the spatial error terms
        eta = np.dot(L1, np.random.randn(n * n, T))
        epsilon = np.dot(L2, np.random.randn(n * n, T))
        print(
            "based on the error term, the best mse we can achieve will be above %.4f"
            % np.mean((eta + epsilon)**2))

        return Z.astype(np.float32), precipitation.astype(np.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        Z = self.Z[idx]  # idx th sample
        X = self.X[idx]
        mask = self.mask[idx]  # the mask for the idx th sample

        # the mask starts with 1, for example 110100, the input is the observed temporal snapshots aggregated together, in this case, it is [Z[0:2, :], Z[3, :]]
        # split the array Z into chuncks following the indexes in the masks where 0 and 1 alternates
        pos = []
        flip = 0
        for i in range(len(mask)):
            if mask[i] == flip:
                pos.append(i)
                flip = 1 - flip

        Z_split = np.split(Z, pos)
        X_split = np.split(X, pos)

        # input is a list of tensors which takes just the odd indices of Z_split
        input_Z = Z_split[::2]
        output = Z_split[1::2]
        input_X = X_split[::2]
        input_for_decoder = X_split[1::2]

        seq_len = [i.shape[0] for i in output]  # the number of sequences the decoder needs to generate for each block is kept in seq_len

        # input for encoder combines input_Z and input_X, input for decoder only has output_X
        input_for_encoder = [np.concatenate([input_Z[i], input_X[i]], axis=1) for i in range(len(input_Z))]

        # here, additional time-invariant input is None
        additional_time_invariant_input = []



        return [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]

class DatasetDstmDecoderWithoutInput(DatasetDstm):
    """Decoder will have no inputs

    """
    def __init__(self,
                 n,
                 T,
                 gamma,
                 l,
                 offset,
                 total,
                 mask,
                 baseline_underlying,
                 baseline_precipitation ):
        super(DatasetDstmDecoderWithoutInput, self).__init__( n,
                                                              T,
                                                              gamma,
                                                              l,
                                                              offset,
                                                              total,
                                                              mask,
                                                              baseline_underlying,
                                                              baseline_precipitation)
    def __getitem__(self, idx):
        [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len] = super().__getitem__(idx)
        input_for_decoder = []

        return [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]


class DatasetDstmEncoderWithoutInput(DatasetDstm):
    """Encoder will have all zeros as inputs

    """
    def __init__(self,
                 n,
                 T,
                 gamma,
                 l,
                 offset,
                 total,
                 mask,
                 baseline_underlying,
                 baseline_precipitation ):
        super(DatasetDstmEncoderWithoutInput, self).__init__( n,
                                                              T,
                                                              gamma,
                                                              l,
                                                              offset,
                                                              total,
                                                              mask,
                                                              baseline_underlying,
                                                              baseline_precipitation)
    def __getitem__(self, idx):
        [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len] = super().__getitem__(idx)

        input_for_encoder = [np.zeros_like(i) for i in input_for_encoder]

        return [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]





class DatasetDstm2:
    """class for generating data from a basic dynamical spatio-temporal model

    Denote:
    Y_t: a vector containing the underlying process of all spatial locations at time t
    X_t: a vector containing the precipitation at all spatial locations at time t
    S_t = X_t + Y_t
    Z_t = S_t + sptial_error_term

    Y_t(s) = \Sum_{x=1}^{n^2} m(s, x|\gamma, offset, l) * S_{t-1}(x) + \eta_{t}, where \eta_{t} \sim Gau(0, R_{1t})
    and m(s, x | \gamma, offset, l) = \gamma * \exp(-(x - s + offset)^2 / l)
    S_t = Y_t + X_t
    Z_t = S_t + \epsilon_t, where \epsilon_t \sim Gau(0, R_{2t})


    Attributes:
          Z: generated dataset
          mask: a vector of T containing 0 and 1s, indicating the missing patterns of the sequence
          baseline_underlying: Y_0(s)
          baseline_precipitation: average precipitation through time

    """

    def __init__(self,
                 n,
                 T,
                 gamma,
                 l,
                 offset,
                 total,
                 mask,
                 baseline_underlying,
                 baseline_precipitation
                 ):
        """
        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :param mask: a numpy vector of T containing 0 and 1s, indicating the missing patterns of the sequence
        :param baseline_underlying: Y_0(s)
        :param baseline_precipitation: average precipitation through time

        """


        self.mask = mask
        self.baseline_underlying = baseline_underlying
        self.baseline_precipitation = baseline_precipitation

        self.Z, self.X = self.prepare_data(n, T, gamma, l, offset, total)




    def prepare_data(self, n, T, gamma, l, offset, total):
        """generate data from the dstm model


        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :return: a tuple of tensors , both of size (total x T x 1 x n x n)
        """



        location_list = []  # create a list storing each spatial location
        for i in range(n):
            for j in range(n):
                location_list.append([i, j])
        location_list = np.array(location_list)

        distance_matrix_list = [
        ]  # create a list, each element stores the pairwise distance between it and every spatial location
        for i in range(n * n):
            dist = np.array([
                np.sqrt(x[0]**2 + x[1]**2)
                for x in location_list[i] - location_list
            ])
            distance_matrix_list.append(dist)
        distance_matrix_list = np.array(distance_matrix_list)

        weights_matrix = gamma * np.exp(
            -(distance_matrix_list + offset)**2 / l
        )  # create a matrix, each row stores the weights matrix between it and every spatial location

        ## normalize the weights
        # def normalize(x):
        #     return x / np.sum(x)
        #
        # weights_matrix = np.apply_along_axis(normalize, 1, weights_matrix)

        # check stability of the evolving process
        w, _ = np.linalg.eig(weights_matrix)
        max_w = np.max(w)
        if max_w == 1 or max_w > 1:
            print("max eigen value is %f" % max_w)
            raise(ValueError("change initial parameters!"))
        else:
            print("max eigen value is %f" % max_w)
            print("valid initial parameters!")

        # error terms with spatial correlation
        sigma_eta = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L1 = np.linalg.cholesky(sigma_eta).transpose()
        sigma_epsilon = 0.1 * np.exp(-np.abs(distance_matrix_list) / 0.1)
        L2 = np.linalg.cholesky(sigma_epsilon).transpose()

        # simulate obs
        Z = np.zeros((n * n, T, total))
        Y = np.zeros((n * n, T, total))
        S = np.zeros((n * n, T, total))
        precipitation = np.random.randn(n * n, T, total) + self.baseline_precipitation


        for i in range(total):
            eta = np.dot(L1, np.random.randn(n * n, T))
            epsilon = np.dot(L2, np.random.randn(n * n, T))
            Y[:, 0, i] = eta[:, 0] + self.baseline_underlying
            S[:, 0, i] = Y[:, 0, i] + precipitation[:, 0, i]
            Z[:, 0, i] = S[:, 0, i] + epsilon[:, 0]

            for t in range(1, T):
                Y[:, t, i] = np.dot(weights_matrix, (S[:, (t - 1),
                                                     i])[:, None]).ravel() + eta[:, t]
                S[:, t, i] = Y[:, t, i] + precipitation[:, t, i]
                Z[:, t, i] = S[:, t, i ] + epsilon[:, t]

        Z = Z.reshape((n, n, T, total))  # convert data to n x n x T x total
        Z = Z[..., None]
        Z = Z.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n

        precipitation = precipitation.reshape((n, n, T, total))  # convert data to n x n x T x total
        precipitation = precipitation[..., None]
        precipitation = precipitation.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n

        # the best we can do is that we fit everything except the spatial error terms
        eta = np.dot(L1, np.random.randn(n * n, T))
        epsilon = np.dot(L2, np.random.randn(n * n, T))
        print(
            "based on the error term, the best mse we can achieve will be above %.4f"
            % np.mean((eta + epsilon)**2))

        return Z.astype(np.float32), precipitation.astype(np.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        Z = self.Z[idx]  # idx th sample
        X = self.X[idx]
        mask = self.mask[idx]  # the mask for the idx th sample

        # the mask starts with 1, for example 110100, the input is the observed temporal snapshots aggregated together, in this case, it is [Z[0:2, :], Z[3, :]]
        # split the array Z into chuncks following the indexes in the masks where 0 and 1 alternates
        pos = []
        flip = 0
        for i in range(len(mask)):
            if mask[i] == flip:
                pos.append(i)
                flip = 1 - flip

        Z_split = np.split(Z, pos)
        X_split = np.split(X, pos)

        # input is a list of tensors which takes just the odd indices of Z_split
        input_Z = Z_split[::2]
        output = Z_split[1::2]
        input_X = X_split[::2]
        input_for_decoder = X_split[1::2]

        # input for encoder combines input_Z and input_X, input for decoder only has output_X
        input_for_encoder = [np.concatenate([input_Z[i], input_X[i]], axis=1) for i in range(len(input_Z))]

        # here, additional time-invariant input is None
        additional_time_invariant_input = []

        seq_len = [i.shape[0] for i in output]  # the number of sequences the decoder needs to generate for each block is kept in seq_len


        return [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]

class DatasetDstm3:
    """class for generating data from a basic dynamical spatio-temporal model
    The high values are transferring from the top left to right down
    Denote:
    Y_t: a vector containing the underlying process of all spatial locations at time t
    Z_t = Y_t + sptial_error_term

    Y_t(s) = \Sum_{x=1}^{n^2} m(s, x | \theta_1, \theta_2, \theta_3, \theta_4) * Y_{t-1}(x) + \eta_{t}, where \eta_{t} \sim Gau(0, R_{1t})
    and m(s, x | \theta_1, \theta_2, \theta_3, \theta_4) = \theta_1 * \exp( - 1 / \theta_2 * [(x1 - \theta_1 - s1)^2 + (x2 - \theta_2 -s2)^2])



    Attributes:
          Z: generated dataset
          mask: a vector of T containing 0 and 1s, indicating the missing patterns of the sequence
          baseline_underlying: Y_0(s)

    """

    def __init__(self,
                 n,
                 T,
                 theta1,
                 theta2,
                 theta3,
                 theta4,
                 total,
                 mask,
                 baseline_underlying
                 ):
        """
        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param theta1: parameter in the linear dstm model
        :param theta2: parameter in the linear dstm model
        :param theta3: parameter in the linear dstm model
        :param theta4: parameter in the linear dstm model
        :param total: total number of training samples
        :param mask: a numpy vector of T containing 0 and 1s, indicating the missing patterns of the sequence
        :param baseline_underlying: a matrix or a list of matrix, representing Y_0(s) of the ith training sample


        """


        self.mask = mask
        
        # if baseline_underlying is a scalar, then first expand it to be a numpy array, and assuming each training sample has the same baseline
        if not hasattr(baseline_underlying, "__len__"):
            self.baseline_underlying = [np.ones(n**2) * baseline_underlying for i in range(total)]
        else:
            if baseline_underlying.ndim == 1:  # assuming each training sample has the same baseline
                self.baseline_underlying = [baseline_underlying for i in range(total)]
            else:
                self.baseline_underlying = baseline_underlying
        


        self.Z = self.prepare_data(n, T, theta1, theta2, theta3, theta4, total)




    def prepare_data(self, n, T, theta1, theta2, theta3, theta4, total):
        """generate data from the dstm model


        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :return: a tuple of tensors , both of size (total x T x 1 x n x n)
        """



        location_list = []  # create a list storing each spatial location
        for i in range(n):
            for j in range(n):
                location_list.append([i, j])
        location_list = np.array(location_list)

        distance_matrix_list = [
        ]  # create a list, each element stores the pairwise distance between it and every spatial location
        for i in range(n * n):
            dist = np.array([
                np.sqrt((x[0] - theta3)**2 + (x[1] - theta4)**2)
                for x in location_list[i] - location_list
            ])
            distance_matrix_list.append(dist)
        distance_matrix_list = np.array(distance_matrix_list)

        weights_matrix = theta1 * np.exp(
            -(distance_matrix_list)**2 / theta2
        )  # create a matrix, each row stores the weights matrix between it and every spatial location

        ## normalize the weights
        # def normalize(x):
        #     return x / np.sum(x)
        #
        # weights_matrix = np.apply_along_axis(normalize, 1, weights_matrix)

        # check stability of the evolving process
        w, _ = np.linalg.eig(weights_matrix)
        max_w = np.max(w)
        if max_w == 1 or max_w > 1:
            print("max eigen value is %f" % max_w)
            raise(ValueError("change initial parameters!"))
        else:
            print("max eigen value is %f" % max_w)
            print("valid initial parameters!")

        # random error terms
        eta = np.random.randn(n * n, T, total) * 0.01

        # simulate obs
        Z = np.zeros((n * n, T, total))
        Y = np.zeros((n * n, T, total))




        for i in range(total):
            Y[:, 0, i] = self.baseline_underlying[i]

           
            for t in range(1, T):
                Y[:, t, i] = np.dot(weights_matrix, (Y[:, (t - 1),
                                                     i])[:, None]).ravel() + eta[:, t, i]

    
                
        # normalization
        scaled_Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        
        # add error term
        for i in range(total):
            for t in range(T):
                Z[:, t, i] = scaled_Y[:, t, i ] + eta[:, t, i]
        

        Z = Z.reshape((n, n, T, total))  # convert data to n x n x T x total
        Z = Z[..., None]
        Z = Z.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n

     



        # the best we can do is that we fit everything except the spatial error terms
        print(
            "based on the error term, the best mse we can achieve will be above %.4f"
            % np.mean(eta**2))

        return Z.astype(np.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        Z = self.Z[idx]  # idx th sample

        mask = self.mask[idx]  # the mask for the idx th sample

        # the mask starts with 1, for example 110100, the input is the observed temporal snapshots aggregated together, in this case, it is [Z[0:2, :], Z[3, :]]
        # split the array Z into chuncks following the indexes in the masks where 0 and 1 alternates
        pos = []
        flip = 0
        for i in range(len(mask)):
            if mask[i] == flip:
                pos.append(i)
                flip = 1 - flip

        Z_split = np.split(Z, pos)


        # input is a list of tensors which takes just the odd indices of Z_split
        input_for_encoder = Z_split[::2]
        output = Z_split[1::2]

        input_for_decoder = []

        # here, additional time-invariant input is None
        additional_time_invariant_input = []

        seq_len = [i.shape[0] for i in output]  # the number of sequences the decoder needs to generate for each block is kept in seq_len


        return [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]


class MovingMNIST:
    """The famous MovingMNIST dataset"""
    def __init__(self,
                 root,
                 n_frames_input,
                 n_frames_output,
                 num_digits,
                 image_size,
                 digit_size,
                 N,
                 transform=None,
                 use_fixed_dataset=False):
        '''if use_fixed_dataset = True, the mnist_test_seq.npy in the root folder will be loaded'''
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
        data = np.zeros(
            (self.n_frames_total, self.image_size_, self.image_size_),
            dtype=np.float32)
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
                data[i, top:bottom,
                left:right] = np.maximum(data[i, top:bottom, left:right],
                                         digit_image)

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
        images = images.reshape(
            (length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape(
            (length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = [output / 255.0]


        input_for_encoder = [input / 255]
        input_for_decoder = []
        additional_time_invariant_input = []
        seq_len = [10]



        out = [idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len]
        return out

    def __len__(self):
        return self.length


def test_MovingMNIST():
    root = '../data'
    n_frames_input = 10
    n_frames_output = 10
    num_digits = 2
    image_size = 64
    digit_size = 28
    N = 200  # total number of samples including training and validation data

    data = MovingMNIST(root, n_frames_input, n_frames_output, num_digits, image_size, digit_size, N)



    one_sample = data[1]
    input_frames = one_sample[1][0]
    one_frame = input_frames[4, ...]

    plot_spatio_temporal_data(input_frames.squeeze())


















def test_dstm():
    # test the simulation for the dstm model
    n = 5
    T = 15
    gamma = 0.3
    l = 1
    offset = 0
    total = 2
    mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]]) == 1
    baseline_underlying = 100
    baseline_precipitation = 4
    data = DatasetDstm(n, T, gamma, l, offset, total, mask, baseline_underlying, baseline_precipitation)
    assert (len(data)) == 2
    one_sample = data[0]
    assert one_sample[0] == 0
    assert one_sample[1][0].shape == (6, 2, 5, 5)  # first tensor in the input for encoder list
    assert one_sample[2][0].shape == (4, 1, 5, 5)  # first tensor in the input for decoder list
    assert one_sample[4][0].shape == (4, 1, 5, 5)  # first tensor in the output list
    assert one_sample[5] == [4, 3]  # the first missing chunk has 4 obs, and the second missing chunk has 3 obs
    assert one_sample[1][1].shape == (1, 2, 5, 5)
    assert one_sample[2][1].shape == (3, 1, 5, 5)
    assert one_sample[1][2].shape == (1, 2, 5, 5)


def test_dstm2():
    # test the simulation for the dstm model
    n = 5
    T = 15
    gamma = 0.3
    l = 1
    offset = 0
    total = 2
    mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]]) == 1
    baseline_underlying = 100
    baseline_precipitation = 4
    data = DatasetDstm2(n, T, gamma, l, offset, total, mask, baseline_underlying, baseline_precipitation)
    assert (len(data)) == 2
    one_sample = data[0]
    assert one_sample[0] == 0
    assert one_sample[1][0].shape == (6, 2, 5, 5)  # first tensor in the input for encoder list
    assert one_sample[2][0].shape == (4, 1, 5, 5)  # first tensor in the input for decoder list
    assert one_sample[4][0].shape == (4, 1, 5, 5)  # first tensor in the output list
    assert one_sample[5] == [4, 3]  # the first missing chunk has 4 obs, and the second missing chunk has 3 obs
    assert one_sample[1][1].shape == (1, 2, 5, 5)
    assert one_sample[2][1].shape == (3, 1, 5, 5)
    assert one_sample[1][2].shape == (1, 2, 5, 5)

def test_dstm3():
    n = 10
    T = 15
    theta1 = 0.5
    theta2 = 1

    theta3 = 1
    theta4 = 1
    total = 2
    mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]]) == 1
    baseline_underlying = np.array([[100, 100, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [100, 100, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).ravel()
    data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)
    assert (len(data)) == 2
    one_sample = data[0]
    assert one_sample[0] == 0
    assert one_sample[1][0].shape == (6, 1, 10, 10)  # first tensor in the input for encoder list

    assert one_sample[4][0].shape == (4, 1, 10, 10)  # first tensor in the output list
    assert one_sample[5] == [4, 3]  # the first missing chunk has 4 obs, and the second missing chunk has 3 obs




if __name__ == "__main__":
    pass



