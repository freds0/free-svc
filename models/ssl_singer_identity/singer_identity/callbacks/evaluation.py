from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
import numpy as np

from singer_identity.utils.core import similarity, roll
import singer_identity.losses as losses

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random


class BaseEvaluationCallback(Callback):
    """ """

    def __init__(
        self,
        log_n_epochs: int = 1,
        on_train: bool = False,
        use_projection: bool = True,
    ) -> None:
        """
        Args:

        """
        super().__init__()
        self.log_n_epochs = log_n_epochs
        self.on_train = on_train
        self.use_projection = use_projection

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> None:
        if self.on_train:
            self.shared_step(trainer, pl_module, batch, batch_idx, "train")
            pl_module.train()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.shared_step(trainer, pl_module, batch, batch_idx, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.shared_step(trainer, pl_module, batch, batch_idx, "")


class HypersphereEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        # if current_epoch % self.log_n_epochs == 0:

        pos_sample1 = batch["clip1"]
        pos_sample2 = batch["clip2"]
        pl_module.eval()
        if self.use_projection:
            z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
            name = "proj"
        else:
            z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
            name = "emb"

        align_loss = losses.align_loss(z1, z2)
        uniform_loss = losses.uniform_loss(z1) / 2
        uniform_loss += losses.uniform_loss(z2) / 2

        step_name = f"/{step_name}" if step_name != "" else step_name
        pl_module.log(
            f"Alignment evaluation {name}{step_name}",
            align_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=z1.shape[0],
        )
        pl_module.log(
            f"Uniformity evaluation {name}{step_name}",
            uniform_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=z1.shape[0],
        )

# Adapted to run in-batch during training
class EEREvaluation(BaseEvaluationCallback):
    """ """

    def __init__(
        self,
        *args,
        use_more_neg=False,
        **kwargs,
    ) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)
        self.use_more_neg = use_more_neg

    def separate_data(self, agg_vec):
        assert len(agg_vec) % 2 == 0
        total_num = len(agg_vec) // 2
        feature1 = agg_vec[:total_num]
        feature2 = agg_vec[total_num:]
        return feature1, feature2

    def eer(self, labels, scores):
        """
        labels: (N,1) value: 0,1 1 same 0 different
        scores: (N,1) value: -1 ~ 1
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        s = interp1d(fpr, tpr)
        a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
        eer = brentq(a, 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)
        return (eer,)

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_n_epochs == 0:
            pos_sample1 = batch["clip1"]
            pos_sample2 = batch["clip2"]
            labels = batch["group_name"]
            pl_module.eval()
            if self.use_projection:
                z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
                name = "proj"
            else:
                z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
                name = "emb"

            # positive scores
            scores = similarity(z1, z2).cpu().detach().tolist()
            labels = [1] * len(scores)
           
            neg = roll(z2)
        
            # Neg scores
            scores = scores + similarity(z1, neg).cpu().detach().tolist()
            labels = labels + ([0] * len(labels))

            eer, *others = self.eer(np.array(labels), np.array(scores))

            step_name = f"/{step_name}" if step_name != "" else step_name
            pl_module.log(
                f"EER evaluation {name}{step_name}",
                eer,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )



# This is the same as Mean Normalized rank adapted to run in-batch during training
class OrderEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_n_epochs == 0:
            pl_module.eval()
            pos_sample1 = batch["clip1"]
            pos_sample2 = batch["clip2"]
            labels = batch["group_name"]
            if self.use_projection:
                z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
                name = "proj"
            else:
                z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
                name = "emb"

            pair_positions = []
            times = 1
            for k in range(times):
                for i, inst_x in enumerate(z1):
                    # Find locations where label is different from anchor label
                    # And places current index at position 0
                    # valid_locations = [i] + [j for j,x in enumerate(labels) if x!=labels[i]]
                    valid_locations = [i] + [j for j, x in enumerate(labels) if i != j]

                    fit = (
                        similarity(inst_x[None, :], z2[valid_locations], 1)
                        .detach()
                        .cpu()
                        .numpy()
                    )  # .cpu().numpy()
                    order = np.argsort(fit)[::-1]
                    anchor_position = np.where(order == 0)[0][0]
                    pair_positions.append(anchor_position / len(valid_locations))
            mean = np.mean(pair_positions)
            var = np.var(pair_positions)
            median = np.median(pair_positions)
            step_name = f"/{step_name}" if step_name != "" else step_name
            pl_module.log(
                f"Order evaluation mean {name}{step_name}",
                mean,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )
            pl_module.log(
                f"Order evaluation median {name}{step_name}",
                median,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )
            pl_module.log(
                f"Order evaluation variance {name}{step_name}",
                var,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )


# Not used in the paper
class KNNEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, n_neighbors=10, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.le = {}
        self.vectors = {}
        self.labels = {}

    # Fit label encoders
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for i, dataloader in enumerate(trainer.test_dataloaders):
            labels = list(trainer.test_dataloaders[i].dataset.groups.keys())
            self.le[i] = LabelEncoder()
            self.le[i].fit(labels)
            self.vectors[i] = []
            self.labels[i] = []

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        (pos_sample1, pos_sample2, more_neg, labels) = batch
        pl_module.eval()

        # z1 = pl_module(pos_sample1.to(pl_module.device))
        # z2 = pl_module(pos_sample2.to(pl_module.device))

        if self.use_projection:
            z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
            name = "proj"
        else:
            z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
            name = "emb"

        self.vectors[dataloader_idx].append(z1.cpu().numpy())
        self.labels[dataloader_idx].append(labels)

        if batch_idx == trainer.num_test_batches[dataloader_idx] - 1:
            X = np.concatenate([np.array(j) for j in self.vectors[dataloader_idx]])
            y = [k for j in self.labels[dataloader_idx] for k in j]
            y = self.le[dataloader_idx].transform(y)
            self.neigh.fit(X, y)
            score = self.neigh.score(X, y)
            pl_module.log(f"KNN score {name}", score)
            self.vectors[dataloader_idx] = []
            self.labels[dataloader_idx] = []

