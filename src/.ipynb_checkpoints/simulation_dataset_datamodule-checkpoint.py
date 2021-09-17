"""pytorch lightning data wrapper for convenient use

"""
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from simulation_dataset import DatasetDstm


class DatasetDataModule(pl.LightningDataModule):
    """organize the data for training, validation and testing in pytorch

    """
    def __init__(self, data, batch_size, training_ratio):
        """

        :param data: Dataset object
        :param batch_size: batch size
        :param training_ratio: the proportion of training data among the whole dataset

        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        N = len(data)
        self.training_data_size = int(np.ceil(N * training_ratio))
        self.validation_data_size = int(N - self.training_data_size)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.training_data, self.validation_data = random_split(self.data, [self.training_data_size,
                                                                                self.validation_data_size])

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

if __name__ == "__main__":
    mask = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) == 1
    data = DatasetDstm(5, 10, 0.2, 1, 0, 100, mask, 0, 0)
    data_module = DatasetDataModule(data, 2, 0.5)
    assert data_module.training_data_size == 50