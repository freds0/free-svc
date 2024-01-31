from math import cos, pi
from typing import Optional, Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class MAWeightUpdate(Callback):
    """Weight update rule from BYOL.
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[MAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996, max_epochs=100, should_update: bool = True):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.max_epochs = max_epochs
        self.should_update = should_update

        self.current_tau = initial_tau

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            unused: Optional[int] = 0
    ) -> None:
        # get networks
        student_network = pl_module.student_network
        teacher_network = pl_module.teacher_network

        # update weights
        self.update_weights(student_network, teacher_network)

        # log tau
        pl_module.log("hparams/MA rate", self.current_tau, prog_bar=False, logger=True)

        # update tau after
        if self.should_update:
            self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * self.max_epochs
        tau = 1 - (1 - self.initial_tau) * (cos(pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(
        self,
        student_network: torch.nn.Module,
        teacher_network: torch.nn.Module
    ) -> None:
        # apply MA weight update
        for (name, student_p), (_, teacher_p) in zip(
            student_network.named_parameters(),
            teacher_network.named_parameters(),
        ):
            teacher_p.data = self.current_tau * teacher_p.data + (1 - self.current_tau) * student_p.data
