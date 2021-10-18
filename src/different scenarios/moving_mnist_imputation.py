%load_ext autoreload
%autoreload 2
import sys

# insert at 1, 0 is the script path (or ‘’ in REPL)
sys.path.insert(1, '../')


import numpy as np
import torch
from torch.utils.data import DataLoader
from bayesian_neural_net import CLSTM_cell
from bayesian_neural_net import ConvCell
from bayesian_neural_net import ConvRelu
from bayesian_neural_net import DeconvRelu
from bayesian_neural_net import Encoder_pro
from bayesian_neural_net import Decoder_pro
from bayesian_neural_net import ED_pro
from bayesian_neural_net import Bidirectional_ED_pro


from simulation_dataset import MovingMNIST2
from simulation_dataset_datamodule import DatasetDataModule
from trainer import LightningED2
from visualization import plot_spatio_temporal_data

import pytorch_lightning as pl

import copy

# set the seed globally for reproducibility
torch.manual_seed(0)
np.random.seed(0)

root = './'
n_frames = 20
num_digits = 2
image_size = 64
digit_size = 28
N = 10 # total number of samples including training and validation data
mask = np.ones((N, n_frames)) * np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
data = MovingMNIST2( root,
                     n_frames,
                     mask,
                     num_digits,
                     image_size,
                     digit_size,
                     N,
                     transform=None,
                     use_fixed_dataset=False)

data_module = DatasetDataModule(data, 1, 0.5)

# build the model
# encoder pro
rnns = [CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64,
                   dropout_rate=0.),
        CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.),
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.) ]

convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.),
             ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.),
             ConvRelu(96, 96, 3, 2, 1, dropout_rate=0.)]

encoder_net = Encoder_pro(rnns, convrelus)

# input for encoder
S = 10
B = 2
input_channels = 1
H = 64
W = 64
input_for_encoder = [torch.randn(B, S, input_channels, H, W)]

# decoder_pro
rnns = [CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.),
        CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96, dropout_rate=0.),
        CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64, dropout_rate=0.)]

deconvrelus = [DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.),
               DeconvRelu(96, 96, 4, 2, 1, dropout_rate=0.),
               DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.)]

cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

# ED_pro net
ED_net1 = ED_pro(encoder_net, decoder_net)
ED_net2 = copy.deepcopy(ED_net1)


# bidirectional_ED_pro
ED_net = Bidirectional_ED_pro(ED_net1, ED_net2)
mc_dropout = 5
learning_rate = 1e-4
max_epoch = 1
model = LightningED2(ED_net, mc_dropout, learning_rate)



# load from checkpoint
try:
    model.load_from_checkpoint(checkpoint_path='moving_mnist_imputation.ckpt', ED=ED_net, mc_dropout=mc_dropout, learning_rate=learning_rate)
except:
    print('fail to load the model')
    pass



#logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')

if torch.cuda.is_available():
    trainer = pl.Trainer(max_epochs=max_epoch, gpus=1)
else:
    trainer = pl.Trainer(max_epochs=max_epoch)
trainer.fit(model, data_module)



# save the checkpoint
trainer.save_checkpoint("moving_mnist_imputation.ckpt")