import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, ' /Users/kehuiyao/Desktop/ConvLSTM-PyTorch')
from torch import nn
from utils import make_layers
import torch
import logging


from BayesianDropout import BayesianDropout

class Encoder(nn.Module):
    """
    used to enocde the data.
    consists of multiple ConvLSTM cells and glue layers.
    """
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        # subnets is a list, rnns is a list, params is a dictionary, rnn is a ConvSLTM obj
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))

            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn, dropout_i=0, dropout_m=0):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)


        # apply the CNN version of variational dropout, this combines the idea from Yarin Gal: A Theoretically Grounded Application of Dropout in Recurrent Neural Networks and Yarin Gal: BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
        x = inputs.new_empty((1, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))

        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))

        dropout = BayesianDropout(dropout_i, x)
        inputs = dropout(inputs)

        outputs_stage, state_stage = rnn(inputs, hidden_state=None, seq_len=10, dropout=dropout_m)
        return outputs_stage, state_stage

    def forward(self, inputs, dropout_i=0, dropout_m=0):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)), dropout_i, dropout_m)
            hidden_states.append(state_stage)
        return tuple(hidden_states)


if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_decoder_params
    from data.mm import MovingMNIST

    encoder = Encoder(convlstm_encoder_params[0],
                      convlstm_encoder_params[1])
    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=4,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        if i == 1:
            break
        inputs = inputVar.to(device)  # B,S,1,64,64
        print(inputs.size())
        state = encoder(inputs)

