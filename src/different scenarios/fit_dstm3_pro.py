import sys
# insert at 1, 0 is the script path (or ‘’ in REPL)
sys.path.insert(1, '../')
import numpy as np
import torch
from torch.utils.data import DataLoader
from bayesian_neural_net import CLSTM_cell
from bayesian_neural_net import ConvCell
from bayesian_neural_net import Encoder_pro
from bayesian_neural_net import Decoder_pro
from bayesian_neural_net import ED_pro
from bayesian_neural_net import ConvRelu
from bayesian_neural_net import DeconvRelu
from visualization import plot_spatio_temporal_data
from simulation_dataset import DatasetDstm
from simulation_dataset import DatasetDstmDecoderWithoutInput
from simulation_dataset import DatasetDstmEncoderWithoutInput
from simulation_dataset import DatasetDstm3
from simulation_dataset_datamodule import DatasetDataModule
from trainer import LightningED
from visualization import plot_spatio_temporal_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# simulate the data
n = 10
T = 15
theta1 = 0.5
theta2 = 1
theta3 = 1
theta4 = 1
total = 200
mask = np.ones([total, T]) * np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) == 1
baseline_underlying = np.random.randn(total, n**2)  # the baseline changes for every sample
data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)
data_module = DatasetDataModule(data, 1, 0.5)


# build the model
# encoder pro
rnns = [CLSTM_cell(shape=(10, 10), input_channels=16, filter_size=3, num_features=64,
                   dropout_rate=0.1),
        CLSTM_cell(shape=(5, 5), input_channels=64, filter_size=5, num_features=96, dropout_rate=0.1)
        ]

convrelus = [ConvRelu(1, 16, 3, 1, 1, dropout_rate=0.1),
             ConvRelu(64, 64, 3, 2, 1, dropout_rate=0.1)
             ]

encoder_net = Encoder_pro(rnns, convrelus)

# input for encoder
S = 10
B = 2
input_channels = 1
H = 64
W = 64
input_for_encoder = [torch.randn(B, S, input_channels, H, W)]

# decoder_pro
rnns = [CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=3, num_features=96, dropout_rate=0.1),
        CLSTM_cell(shape=(10, 10), input_channels=64, filter_size=3, num_features=64, dropout_rate=0.1)
        ]

deconvrelus = [DeconvRelu(96, 64, 2, 2, 0, dropout_rate=0.1),
               DeconvRelu(64, 16, 3, 1, 1, dropout_rate=0.1)
               ]


cnn = ConvCell(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
decoder_net = Decoder_pro(rnns, deconvrelus, cnn)

# ED net
ED_net = ED_pro(encoder_net, decoder_net)
mc_dropout = 5
learning_rate = 1e-4
max_epoch = 20
model = LightningED(ED_net, mc_dropout, learning_rate)

# load from checkpoint
# try:
#     model.load_from_checkpoint(checkpoint_path='different_baseline_values_for_each_training_sample.ckpt', ED=ED_net, mc_dropout=mc_dropout, learning_rate=learning_rate)
# except:
#     pass

#logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')

if torch.cuda.is_available():
    trainer = pl.Trainer(max_epochs=max_epoch, gpus=1)
else:
    trainer = pl.Trainer(max_epochs=max_epoch)
trainer.fit(model, data_module)

# save the checkpoint
trainer.save_checkpoint("temp.ckpt")