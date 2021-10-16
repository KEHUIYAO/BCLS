import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler




class LightningED(pl.LightningModule):
    """Pytorch lightning training process for unidirectional convlstm
    """

    def __init__(self, ED, mc_dropout,
                 learning_rate=1e-3):
        """

        :param ED: an ED network object
        :param mc_dropout: number of stochastic forward passes
        :param learning_rate: learning rate
        """
        super(LightningED, self).__init__()
        self.ED = ED
        self.loss_function = nn.MSELoss()
        self.mc_dropout = mc_dropout
        self.learning_rate = learning_rate

    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len):

        return self.ED(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)

    def training_step(self, batch, batch_idx):
        idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len = batch


        pred = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)  # (B,S,C,H,W)

        output = torch.cat(output, dim=1)
        pred = torch.cat(pred, dim=1)
        loss = self.loss_function(pred, output)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        "the function is called after every epoch is completed"

        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        #print(avg_loss)


        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len = batch
        res = []
        for i in range(self.mc_dropout):
            pred = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)  # B,S,C,H,W
            pred = torch.cat(pred, dim=1)
            res.append(pred)
        pred_avg = torch.stack(res).mean(dim=0)
        pred_std = torch.stack(res).std(dim=0)
        output = torch.cat(output, dim=1)
        loss = self.loss_function(pred_avg, output)
        self.log('validation_loss', loss)

        naive_predictor = torch.ones_like(output).permute(3, 4, 0, 1, 2) * output.mean(dim=(-1, -2))
        naive_predictor = naive_predictor.permute(2, 3, 4, 0, 1)
        naive_predictor_loss = self.loss_function(naive_predictor, output)

        return {'loss': loss, 'naive_predictor_loss': naive_predictor_loss}

    def validation_epoch_end(self, outputs):
        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_naive_predictor_loss = torch.stack([x['naive_predictor_loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Loss/Naive_predictor', avg_naive_predictor_loss, self.current_epoch)
        self.log('val_loss', avg_loss)  # metric to be tracked
        
        # print(avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val_loss'}}


class LightningED2(pl.LightningModule):
    """Pytorch lightning training process for bidirectional convlstm
    """

    def __init__(self, bidirectional_ED_pro , mc_dropout,
                 learning_rate=1e-3):
        """

        :param ED: an ED network object
        :param mc_dropout: number of stochastic forward passes
        :param learning_rate: learning rate
        """
        super(LightningED2, self).__init__()
        self.model = bidirectional_ED_pro
        self.loss_function = nn.MSELoss()
        self.mc_dropout = mc_dropout
        self.learning_rate = learning_rate

    def forward(self, input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len):

        return self.model(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)

    def training_step(self, batch, batch_idx):
        idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len = batch

        # prediction by the forward pass and prediction by the backward pass
        pred1, pred2 = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)  # (B,S,C,H,W)

        output = torch.cat(output, dim=1)
        pred1 = torch.cat(pred1, dim=1)
        pred2 = torch.cat(pred2, dim=1)
        loss1 = self.loss_function(pred1, output)
        loss2 = self.loss_function(pred2, output)

        # discrepancy loss
        loss3 = self.loss_function(pred1, pred2)

        # total loss
        loss = loss1 + loss2 + loss3
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        "the function is called after every epoch is completed"

        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        #print(avg_loss)


        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        idx, input_for_encoder, input_for_decoder, additional_time_invariant_input, output, seq_len = batch
        res1 = []
        res2 = []
        for i in range(self.mc_dropout):
            pred1, pred2 = self.forward(input_for_encoder, input_for_decoder, additional_time_invariant_input, seq_len)  # B,S,C,H,W
            pred1 = torch.cat(pred1, dim=1)
            pred2 = torch.cat(pred2, dim=1)
            res1.append(pred1)
            res2.append(pred2)
        pred_avg1 = torch.stack(res1).mean(dim=0)
        pred_std1 = torch.stack(res1).std(dim=0)
        pred_avg2 = torch.stack(res2).mean(dim=0)
        pred_std2 =  torch.stack(res2).std(dim=0)
        output = torch.cat(output, dim=1)
        loss1 = self.loss_function(pred_avg1, output)
        loss2 = self.loss_function(pred_avg2, output)
        loss3 = self.loss_function(pred_avg1, pred_avg2)

        loss = loss1 + loss2 + loss3
        self.log('validation_loss', loss)

        naive_predictor = torch.ones_like(output).permute(3, 4, 0, 1, 2) * output.mean(dim=(-1, -2))
        naive_predictor = naive_predictor.permute(2, 3, 4, 0, 1)
        naive_predictor_loss = self.loss_function(naive_predictor, output)

        return {'loss': loss, 'naive_predictor_loss': naive_predictor_loss}

    def validation_epoch_end(self, outputs):
        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_naive_predictor_loss = torch.stack([x['naive_predictor_loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Loss/Naive_predictor', avg_naive_predictor_loss, self.current_epoch)
        self.log('val_loss', avg_loss)  # metric to be tracked

        # print(avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val_loss'}}
