import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from hydra.utils import to_absolute_path
# Globals
MEANS = {
    "CIFAR10": (0.4914, 0.4822, 0.4465),
}
STDS = {
    "CIFAR10": (0.2470, 0.2435, 0.2616),
}


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, trfs=None):
        super().__init__()
        self.name = cfg.data.name
        self.root = to_absolute_path(cfg.data.root)
        logger.debug(self.root)
        self.loader_params = cfg.data.loader_params
        # Transforms
        means, stds = MEANS[self.name], STDS[self.name]
        if trfs is not None:
            self.train_transforms = trfs
            self.test_transforms = trfs
        else:
            self.train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4), # standard DA
                transforms.RandomHorizontalFlip(), # standard DA
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
            self.test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

    def prepare_data(self):
        # debug

        # download data if needed
        datasets.cifar.CIFAR10(
            root=self.root,
            train=True,
            download=True
        )
        datasets.cifar.CIFAR10(
            root=self.root,
            train=False,
            download=True
        )

    def setup(self, stage=None):
        # Assign train/val/test for dataloaders
        train_data = datasets.cifar.CIFAR10(
            root=self.root,
            train=True,
            download=False,
            transform=self.train_transforms,
        )
        test_data = datasets.cifar.CIFAR10(
            root=self.root,
            train=False,
            download=False,
            transform=self.test_transforms,
        )

        self.train_data = train_data
        self.val_data = test_data
        self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, **self.loader_params)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.loader_params)
