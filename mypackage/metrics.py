# Imports needed for MyAccuracy
from typing import Any, Callable, Optional
from pytorch_lightning.metrics import Metric
import pytorch_lightning as pl
if pl.__version__.startswith("1.1"):
    # PTL 1.1.X
    from pytorch_lightning.metrics.utils import _input_format_classification
import torch

class MyAccuracy(Metric):
    def __init__(
            self,
            threshold: float = 0.5,
            compute_on_step: bool = True,
            dist_sync_on_step=False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        if pl.__version__ == "1.0.5":
            super().__init__(
                compute_on_step=compute_on_step,
                dist_sync_on_step=dist_sync_on_step,
                process_group=process_group,
            )
        else:   
            super().__init__(
                compute_on_step=compute_on_step,
                dist_sync_on_step=dist_sync_on_step,
                process_group=process_group,
                dist_sync_fn=dist_sync_fn,
            )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
            raise ValueError(
                "preds and target must have same number of dimensions, or one additional dimension for preds"
            )

        if len(preds.shape) == len(target.shape) + 1:
            # multi class probabilites
            preds = torch.argmax(preds, dim=1)

        if len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
            # binary or multilabel probablities
            preds = (preds >= self.threshold).long()
        return preds, target
    
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        # PTL 1.1.X
        if pl.__version__.startswith("1.1"):
            preds, target = _input_format_classification(logits, target, self.threshold)
        else:
            preds, target = self._input_format(logits, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return 100. * self.correct / self.total
