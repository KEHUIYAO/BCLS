import numpy as np
import torch
from torch.utils.data import DataLoader
from bayesian_neural_net import CLSTM_cell
from bayesian_neural_net import ConvCell
from bayesian_neural_net import Encoder
from bayesian_neural_net import Decoder
from bayesian_neural_net import ED
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

def test_dstm_decoder_with_inputs():
    n = 5
    T = 15
    gamma = 0.3
    l = 1
    offset = 0
    total = 200
    mask = np.ones([total, T]) * np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]) == 1

    baseline_underlying = 10
    baseline_precipitation = 4
    data = DatasetDstm(n, T, gamma, l, offset, total, mask, baseline_underlying, baseline_precipitation)
    data_module = DatasetDataModule(data, 1, 0.5)

    rnns = [CLSTM_cell(shape=(5, 5), input_channels=2, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)

    mc_dropout = 5
    learning_rate = 1e-3
    max_epoch = 20
    model = LightningED(ED_net, mc_dropout, learning_rate)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    # trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epoch)



    # trainer.tune(model, dm)
    trainer.fit(model, data_module)

def test_dstm_decoder_without_inputs():
    n = 5
    T = 15
    gamma = 0.3
    l = 1
    offset = 0
    total = 200
    mask = np.ones([total, T]) * np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]) == 1

    baseline_underlying = 10
    baseline_precipitation = 4
    data = DatasetDstmDecoderWithoutInput(n, T, gamma, l, offset, total, mask, baseline_underlying, baseline_precipitation)
    data_module = DatasetDataModule(data, 1, 0.5)

    rnns = [CLSTM_cell(shape=(5, 5), input_channels=2, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)

    mc_dropout = 5
    learning_rate = 1e-3
    max_epoch = 20
    model = LightningED(ED_net, mc_dropout, learning_rate)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    # trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epoch)



    # trainer.tune(model, dm)
    trainer.fit(model, data_module)

def test_dstm_encoder_without_inputs():
    n = 5
    T = 15
    gamma = 0.3
    l = 1
    offset = 0
    total = 200
    mask = np.ones([total, T]) * np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]) == 1

    baseline_underlying = 10
    baseline_precipitation = 4
    data = DatasetDstmEncoderWithoutInput(n, T, gamma, l, offset, total, mask, baseline_underlying, baseline_precipitation)
    data_module = DatasetDataModule(data, 1, 0.5)

    rnns = [CLSTM_cell(shape=(5, 5), input_channels=2, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(5, 5), input_channels=1, filter_size=1, num_features=16, dropout_rate=0.5), CLSTM_cell(shape=(5, 5), input_channels=16, filter_size=1, num_features=16, dropout_rate=0.5)]
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)

    mc_dropout = 5
    learning_rate = 1e-3
    max_epoch = 20
    model = LightningED(ED_net, mc_dropout, learning_rate)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    # trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epoch)



    # trainer.tune(model, dm)
    trainer.fit(model, data_module)


def test_dstm3():
    n = 10
    T = 15
    theta1 = 0.5
    theta2 = 1

    theta3 = 1
    theta4 = 1
    total = 200
    mask = np.ones([total, T]) * np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 1
    baseline_underlying = np.array([[10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).ravel()
    data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)
    data_module = DatasetDataModule(data, 1, 0.5)

    rnns = [CLSTM_cell(shape=(10, 10), input_channels=1, filter_size=3, num_features=16, dropout_rate=0.2), CLSTM_cell(shape=(10, 10), input_channels=16, filter_size=3, num_features=16, dropout_rate=0.2)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(10, 10), input_channels=1, filter_size=3, num_features=16, dropout_rate=0.2), CLSTM_cell(shape=(10, 10), input_channels=16, filter_size=3, num_features=16, dropout_rate=0.2)]
    cnn = ConvCell(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)

    mc_dropout = 5
    learning_rate = 1e-3
    max_epoch = 2
    model = LightningED(ED_net, mc_dropout, learning_rate)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    # trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epoch)



    # trainer.tune(model, dm)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # test_dstm_decoder_without_inputs()
    # test_dstm_decoder_with_inputs()
    # test_dstm_encoder_without_inputs()
    # test_dstm3()
    # tensorboard --logdir tb_logs/Bayesian_ConvLSTM

    n = 10
    T = 15
    theta1 = 0.5
    theta2 = 1

    theta3 = 1
    theta4 = 1
    total = 200
    mask = np.ones([total, T]) * np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) == 1
    baseline_underlying = np.array([[10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).ravel()
    data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)
    data_module = DatasetDataModule(data, 1, 0.5)

    rnns = [CLSTM_cell(shape=(10, 10), input_channels=1, filter_size=3, num_features=64, dropout_rate=0.2), CLSTM_cell(shape=(10, 10), input_channels=64, filter_size=3, num_features=64, dropout_rate=0.2)]
    encoder_net = Encoder(rnns)
    rnns = [CLSTM_cell(shape=(10, 10), input_channels=1, filter_size=3, num_features=64, dropout_rate=0.2), CLSTM_cell(shape=(10, 10), input_channels=64, filter_size=3, num_features=64, dropout_rate=0.2)]
    cnn = ConvCell(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
    decoder_net = Decoder(rnns, cnn)
    ED_net = ED(encoder_net, decoder_net)

    mc_dropout = 5
    learning_rate = 1e-3
    max_epoch = 10
    model = LightningED(ED_net, mc_dropout, learning_rate)
    logger = TensorBoardLogger('tb_logs', name='Bayesian_ConvLSTM')
    # trainer = pl.Trainer(auto_lr_find=True, logger=logger)
    # trainer = pl.Trainer(logger=logger, fast_dev_run=True)
    # trainer = pl.Trainer(logger=logger, max_epochs=args.max_epoch, gpus=1)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epoch)



    # trainer.tune(model, dm)
    trainer.fit(model, data_module)

    model.eval()
    baseline_underlying = np.array([[10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).ravel()
    test_data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, 1, mask, baseline_underlying)
    test_data_loader = DataLoader(test_data, 1)  # batch size = 1

    # predict for one sample
    for idx, batch in enumerate(test_data_loader):

        idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len = batch
        output = model(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)
        if idx == 0:
            break

    # plot the true data
    true_data = test_data.Z[0, ...].squeeze(1)
    plot_spatio_temporal_data(true_data)
    predicted_data = torch.cat(output, dim=1).detach().numpy().squeeze(0).squeeze(1)
    plot_spatio_temporal_data(predicted_data)




