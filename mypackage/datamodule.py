import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        # TODO: Do stuff here
        self.size = size
        self.values = torch.rand((size, 3, 32, 32))
        self.labels = torch.randint(low=0, high=10, size=(size,))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.loader_params = cfg.data.loader_params
        self.size = cfg.data.size

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        data = MyDataset(self.size)
        self.train_data = data
        self.val_data = data
        self.test_data = data

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, **self.loader_params)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.loader_params)
