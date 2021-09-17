from torch import nn
from utils import make_layers
import torch
from BayesianDropout import BayesianDropout


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn, dropout_i=0, dropout_m=0):
        inputs, state_stage = rnn(inputs, state, seq_len=10, dropout=dropout_m)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)

        x = inputs.new_empty((1, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))

        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))

        dropout = BayesianDropout(dropout_i, x)
        inputs = dropout(inputs)


        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states, dropout_i=0, dropout_m=0):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage1'),
                                       getattr(self, 'rnn1'), dropout_i, dropout_m)
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_decoder_params
    from data.mm import MovingMNIST
    from encoder import Encoder
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(convlstm_encoder_params[0],
                      convlstm_encoder_params[1]).to(device)
    decoder = Decoder(convlstm_decoder_params[0],
                      convlstm_decoder_params[1]).to(device)
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=8,
        shuffle=False,
    )

    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,1,64,64
        state = encoder(inputs)
        break
    output = decoder(state)
    print(output.shape)  # B,S,1,64,64
