from typing import Optional, Any

import torch
import torch.nn as nn
from singer_identity.models.byol import TeacherStudentModel, Optimizer, Scheduler
from singer_identity.model import IdentityEncoder, Projection, SiameseArm, MLP

import copy

class BYOL(TeacherStudentModel):
    def __init__(
        self,
        # module: nn.Module,
        weight_callback,
        optimizer: Optimizer,
        backbone: dict = {},
        projection: dict  = {},
        predictor: dict = {},
        feature_extractor: dict = {},
        loss_fn: nn.Module = torch.nn.MSELoss(),
        scheduler: Optional[Scheduler] = None,
        normalize_projections: bool = False,
        normalize_representations: bool = False,
    ):
        encoder = IdentityEncoder(feature_extractor=feature_extractor, encoder=backbone)
        projection_layer = Projection(**projection)
        predictor_layer = MLP(**copy.deepcopy(predictor))
        module = SiameseArm(
            encoder=encoder,
            projector=projection_layer,
            predictor=predictor_layer,
            normalize_projections=normalize_projections,
            normalize_representations=normalize_representations,
        )

        super(BYOL, self).__init__(
            module, loss_fn, weight_callback, optimizer, scheduler=scheduler
        )
        self.save_hyperparameters(ignore=["module", "loss_fn"])

    def shared_step(self, batch, step_name: str):
        x1 = batch["clip1"]
        x2 = batch["clip2"]

        batch_size = x1.shape[0]

        ys, zs, qs = self.student_network(x1)
        with torch.no_grad():
            yt, zt, qt = self.teacher_network(x2)
        loss_12 = self.loss_fn(qs, zt)

        ys, zs, qs = self.student_network(x2)
        with torch.no_grad():
            yt, zt, qt = self.teacher_network(x1)
        loss_21 = self.loss_fn(qs, zt)

        loss = (loss_12 + loss_21) / 2

        self.log(
            f"loss/{step_name}",
            loss,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.record_variables(y1=ys, z1=zs, y2=yt, z2=zt)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
