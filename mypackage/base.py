# Imports needed for LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .metrics import MyAccuracy

class LitClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.best_val_acc = torch.tensor(0.)
        self.train_accuracy = MyAccuracy()
        self.val_accuracy = MyAccuracy()
        self.test_accuracy = MyAccuracy()

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(outputs, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)

        self.log("train_acc_s", self.train_accuracy(out, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        val_loss = self.loss(out, y)
        self.log("val_loss", val_loss)

        results = {"val_acc": self.val_accuracy(out, y)}
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        self.log("test_acc_metric", self.test_accuracy(out, y))

    def validation_epoch_end(self, outputs):

        val_acc = self.val_accuracy.compute()

        if self.best_val_acc < val_acc.cpu():
            self.best_val_acc = val_acc.cpu()
            print(f"New best val acc: {self.best_val_acc:.2f}")

        self.log("val_acc", val_acc, prog_bar=True)
        self.log("best_val_acc", self.best_val_acc, prog_bar=True)

    def test_epoch_end(self, outputs):

        self.log("test_acc_metric", self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return [optimizer], [scheduler]
