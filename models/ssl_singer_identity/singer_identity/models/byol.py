import abc
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

import pytorch_lightning as pl
from copy import deepcopy

try:
    from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
except ModuleNotFoundError:
    print("No Lightning Bolts")


class Optimizer(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.optimizer_class = None

    def __call__(self, parameters) -> torch.optim.Optimizer:
        return self.optimizer_class(parameters, *self.args, **self.kwargs)

    def __str__(self):
        params = "\n".join(f"\t{arg}," for arg in self.args) + "\n".join(
            f"\t{k}: {v}" for k, v in self.kwargs.items()
        )
        return self.optimizer_class.__name__ + "(\n" + params + "\n)"


class Scheduler(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.scheduler_class = None

    def __call__(self, optimizer):
        return self.scheduler_class(optimizer, *self.args, **self.kwargs)

    def __str__(self):
        params = "\n".join(f"\t{arg}," for arg in self.args) + "\n".join(
            f"\t{k}: {v}" for k, v in self.kwargs.items()
        )
        return self.scheduler_class.__name__ + "(\n" + params + "\n)"


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        **kwargs,
    ):
        super(Adam, self).__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )
        self.optimizer_class = torch.optim.Adam


class LinearWarmupCosineAnnealing(Scheduler):
    def __init__(
        self,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        **kwargs,
    ) -> None:
        super(LinearWarmupCosineAnnealing, self).__init__(
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min,
            last_epoch=last_epoch,
            **kwargs,
        )
        self.scheduler_class = LinearWarmupCosineAnnealingLR


class SSLModel(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        module: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Scheduler] = None,
    ):
        super(SSLModel, self).__init__()
        self.module = module
        self.loss_fn = loss_fn
        self.records = {}

        self.use_wandb = None

        self.predictions = None

    def on_fit_start(self) -> None:
        # if not dist.is_initialized() or dist.get_rank() == 0:  # TODO: figure out why it makes everything freeze
        # print(self.module)
        self.print_hyperparameters()

        if hasattr(self, "loss_weighting"):
            self.trainer.callbacks.append(self.loss_weighting)

    def on_train_start(self) -> None:
        try:
            if self.trainer.current_epoch == 0:
                self.save_initial_model()
        except AttributeError:
            warnings.warn("Could not save the initial model")

    def forward(self, x: torch.Tensor):
        y, _, _ = self.module(x)
        return y

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y, z, q = self.module(x)
        return {"Representations": y, "Projections": z, "Predictions": q}

    def on_predict_start(self) -> None:
        self.predictions = {}

    def on_predict_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        _, labels = batch

        keys = list(labels.keys()) if isinstance(labels, dict) else ["labels"]
        keys += ["representations", "projections"]

        self.predictions[dataloader_idx] = {k: [] for k in keys}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, labels = batch
        x = x[0]
        y, z, _ = self.module(x)
        self.predictions[dataloader_idx]["representations"].append(y.cpu())
        self.predictions[dataloader_idx]["projections"].append(z.cpu())

        if not isinstance(labels, dict):
            labels = {"labels": labels}
        for k, v in labels.items():
            self.predictions[dataloader_idx][k].append(v.cpu())

    def on_predict_end(self) -> None:
        for idx, predictions in self.predictions.items():
            predictions = {k: torch.cat(v) for k, v in predictions.items()}
            if dist.is_initialized():
                for key, pred in predictions.items():
                    # get number of predictions for each rank
                    num_predictions_local = torch.tensor(
                        pred.size(0), dtype=torch.long, device=self.device
                    )

                    # get the highest number of predictions
                    max_num_predictions = num_predictions_local.clone()
                    dist.all_reduce(max_num_predictions, op=dist.ReduceOp.MAX)

                    # pad predictions so that it's the same shape across devices
                    padding_size = list(pred.size())
                    padding_size[0] = max_num_predictions - num_predictions_local
                    pred = torch.cat(
                        (pred, torch.zeros(*padding_size, dtype=pred.dtype)), dim=0
                    ).to(self.device)

                    if self.trainer.global_rank == 0:
                        world_size = dist.get_world_size()
                        # get the number of predictions per device
                        num_predictions_per_device = [
                            torch.zeros_like(num_predictions_local)
                            for _ in range(world_size)
                        ]
                        dist.gather(
                            num_predictions_local, num_predictions_per_device, dst=0
                        )

                        # retrieve the padded predictions from all devices
                        all_predictions = [
                            torch.zeros_like(pred) for _ in range(world_size)
                        ]
                        dist.gather(pred, all_predictions, dst=0)

                        # unpad and concatenate the predictions from all devices
                        predictions[key] = torch.cat(
                            [
                                p[:n].cpu()
                                for n, p in zip(
                                    num_predictions_per_device, all_predictions
                                )
                            ]
                        )
                    else:
                        dist.gather(num_predictions_local, dst=0)
                        dist.gather(pred, dst=0)

            self.predictions[idx] = predictions

    def record_variables(self, **kwargs):
        self.records.update({k: v.detach() for k, v in kwargs.items()})

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::conda

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        """
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )

    def save_initial_model(self):
        self.trainer.save_checkpoint(
            self.trainer.checkpoint_callback.format_checkpoint_name(
                dict(epoch=0, step=0)
            )
        )

    @staticmethod
    def instantiate_optimizer(
        parameters, optimizer_cls: Optimizer, scheduler_cls: Optional[Scheduler] = None
    ) -> Dict:
        # instantiate optimizer
        optimizer = optimizer_cls(parameters)
        monitor = dict(optimizer=optimizer)

        # eventually instantiate scheduler
        if scheduler_cls is not None:
            monitor["lr_scheduler"] = scheduler_cls(optimizer)

        return monitor

    def configure_optimizers(self):
        return self.instantiate_optimizer(
            self.module.parameters(), self.hparams.optimizer, self.hparams.scheduler
        )

    # utils
    def print_hyperparameters(self):
        print()
        print(f"Saving logs in {self.trainer.log_dir}...")
        print("Hyperparameters")
        print("---------------")
        for k, v in self.hparams.items():
            print(f"{k}:\t{v}")
        print()


class TeacherStudentModel(SSLModel):
    def __init__(
        self,
        module: nn.Module,
        loss_fn: nn.Module,
        weight_callback,
        optimizer: Optimizer,
        scheduler: Optional[Scheduler] = None,
    ):
        super(TeacherStudentModel, self).__init__(
            module, loss_fn, optimizer, scheduler=scheduler
        )
        self.student_network = self.module
        self.teacher_network = deepcopy(self.student_network)
        for _, p in self.teacher_network.named_parameters():
            p.requires_grad = False
        self.weight_callback = weight_callback

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx
        )
