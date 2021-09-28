import sys

# insert at 1, 0 is the script path (or ‘’ in REPL)
sys.path.insert(1, '../')

import numpy as np
import torch
from torch.utils.data import DataLoader
from bayesian_neural_net import CLSTM_cell
from bayesian_neural_net import ConvCell
from bayesian_neural_net import Encoder
from bayesian_neural_net import Decoder
from bayesian_neural_net import ED

from simulation_dataset import MovingMNIST
from simulation_dataset_datamodule import DatasetDataModule
from trainer import LightningED

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# set the seed globally for reproducibility
torch.manual_seed(0)
np.random.seed(0)

root = './'
n_frames_input = 10
n_frames_output = 10
num_digits = 2
image_size = 64
digit_size = 28
N = 10000  # total number of samples including training and validation data

#data = MovingMNIST(root, n_frames_input, n_frames_output, num_digits, image_size, digit_size, N, use_fixed_dataset=True)
data = MovingMNIST(root, n_frames_input, n_frames_output, num_digits, image_size, digit_size, N, use_fixed_dataset=False)


data_module = DatasetDataModule(data, 1, 0.5)

# build the model
rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=3, num_features=64, dropout_rate=0.1), CLSTM_cell(shape=(64, 64), input_channels=64, filter_size=3, num_features=64, dropout_rate=0.1)]
encoder_net = Encoder(rnns)
rnns = [CLSTM_cell(shape=(64, 64), input_channels=1, filter_size=3, num_features=64, dropout_rate=0.1), CLSTM_cell(shape=(64, 64), input_channels=64, filter_size=3, num_features=64, dropout_rate=0.1)]
cnn = ConvCell(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
decoder_net = Decoder(rnns, cnn)
ED_net = ED(encoder_net, decoder_net)
mc_dropout = 5
learning_rate = 1e-4
max_epoch = 1
model = LightningED(ED_net, mc_dropout, learning_rate)



# load from checkpoint
# try:
#     model.load_from_checkpoint(checkpoint_path='moving_mnist.ckpt', ED=ED_net, mc_dropout=mc_dropout, learning_rate=learning_rate)
# except:
#     pass



#logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')

if torch.cuda.is_available():
    trainer = pl.Trainer(max_epochs=max_epoch, gpus=1)
else:
    trainer = pl.Trainer(max_epochs=max_epoch)         
trainer.fit(model, data_module)



# save the checkpoint
trainer.save_checkpoint("moving_mnist.ckpt")
