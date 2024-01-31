import torch
import torch.nn as nn
from pytorch_lightning.cli import instantiate_class
import pytorch_lightning as pl

import numpy as np

import singer_identity.losses as losses
from singer_identity.model import FeatureExtractor, Encoder, Projection


class SSLTrainer(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: dict = {},
        backbone: dict = {},
        projection: dict = {},
        optimizer1_init: dict = {},
        use_contrastive_loss: bool = False,
        temp: float = 0.1,
        nr_negative: int = 64,
        decouple: bool = False,
        use_invariance_loss: bool = False,
        fact_inv_loss: float = 1,
        use_covariance_reg: bool = False,
        fact_cov: float = 1,
        use_variance_reg: bool = False,
        fact_var: float = 1,
        gamma: float = 1,
        use_vicreg_loss: bool = False,
        use_align_loss: bool = False,
        fact_align_loss: float = 0.25,
        fact_unif_loss: float = 0.5,
        use_uniform_loss: bool = False,
        compute_test_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = FeatureExtractor(**feature_extractor)
        self.encoder = Encoder(**backbone)
        self.projection = Projection(**projection)
        self.net = nn.Sequential(self.feature_extractor, self.encoder, self.projection)

        self.optimizer1_init = optimizer1_init
        self.gamma = (
            gamma
            if not self.projection.l2_normalize
            else (1 / np.sqrt(self.projection.output_dim))
        )
        self.temp = temp
        self.nr_negative = nr_negative
        self.fact_inv_loss = fact_inv_loss
        self.fact_cov = fact_cov
        self.fact_var = fact_var
        self.fact_align_loss = fact_align_loss
        self.fact_unif_loss = fact_unif_loss
        self.decouple = decouple

        self.use_contrastive_loss = use_contrastive_loss
        self.use_vicreg_loss = use_vicreg_loss
        self.use_invariance_loss = use_invariance_loss
        self.use_covariance_reg = use_covariance_reg
        self.use_variance_reg = use_variance_reg
        self.use_align_loss = use_align_loss
        self.use_uniform_loss = use_uniform_loss
        self.reloaded = True
        self.compute_test_loss = compute_test_loss
        print("Declaring trainer")

    def forward(self, x):
        # features = self.feature_extractor(x)
        # feature_embeddings = self.encoder(features)
        # projection = self.projection(feature_embeddings)
        # return projection
        return self.net(x)

    def encode(self, x):
        acoustic_features = self.feature_extractor(x)
        feature_embeddings = self.encoder(acoustic_features)
        return feature_embeddings

    def configure_optimizers(self):
        params = list(self.feature_extractor.parameters())
        params += list(self.encoder.parameters())
        params += list(self.projection.parameters())
        optimizer1 = instantiate_class(params, self.optimizer1_init)
        return optimizer1

    def shared_step(self, batch, batch_idx, step_name, sync_dist=True):
        step_name = f"/{step_name}" if step_name != "" else step_name
        pos_sample1 = batch["clip1"]
        pos_sample2 = batch["clip2"]

        batch_size = pos_sample1.shape[0]

        # Projections of positive pairs
        z1 = self(pos_sample1)
        z2 = self(pos_sample2)

        loss = torch.tensor(0).type_as(z1)
        cont_loss = torch.tensor(0).type_as(z1)
        ratio = 0
        vicreg_loss = torch.tensor(0).type_as(z1)
        inv_loss = torch.tensor(0).type_as(z1)
        cov_reg = torch.tensor(0).type_as(z1)
        var_reg = torch.tensor(0).type_as(z1)
        align_loss = torch.tensor(0).type_as(z1)
        uniform_loss = torch.tensor(0).type_as(z1)

        if self.use_contrastive_loss:
            cont_loss, ratio = losses.contrastive_loss(
                z1,
                z2,
                temp=self.temp,
                nr_negative=self.nr_negative,
                decouple=self.decouple,
            )
            loss += cont_loss
            self.log(f"loss_contrastive{step_name}", cont_loss, batch_size=batch_size)
            self.log(
                f"ratio_contrastive_pos_neg{step_name}", ratio, batch_size=batch_size
            )

        if self.use_vicreg_loss:
            vicreg_loss = vicreg_loss(
                z1,
                z2,
                gamma=self.gamma,
                fact_inv_loss=self.fact_inv_loss,
                fact_var=self.fact_var,
                fact_cov=self.fact_cov,
            )
            loss += vicreg_loss
            self.log(f"loss_vicreg{step_name}", vicreg_loss, batch_size=batch_size)
        else:
            if self.use_invariance_loss:
                inv_loss = losses.invariance_loss(z1, z2) * self.fact_inv_loss
                loss = loss + inv_loss
                self.log(f"loss_invariance{step_name}", inv_loss, batch_size=batch_size)
            if self.use_covariance_reg:
                cov_reg = losses.covariance_reg(z1, z2) * self.fact_cov
                loss += cov_reg
                self.log(f"reg_covariance{step_name}", cov_reg, batch_size=batch_size)
            if self.use_variance_reg:
                var_reg = losses.variance_hinge_reg(z1, z2, self.gamma) * self.fact_var
                loss = loss + var_reg
                self.log(f"reg_variance{step_name}", var_reg, batch_size=batch_size)
        if self.use_align_loss:
            align_loss = losses.align_loss(z1, z2) * self.fact_align_loss
            loss += align_loss
            self.log(f"loss_align{step_name}", align_loss, batch_size=batch_size)
        if self.use_uniform_loss:
            uniform_loss = (
                (losses.uniform_loss(z1) + losses.uniform_loss(z2))
                * self.fact_unif_loss
                / 2
            )
            loss += uniform_loss
            self.log(f"loss_uniform{step_name}", uniform_loss, batch_size=batch_size)

        self.log(f"loss{step_name}", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        if self.compute_test_loss:
            return self.shared_step(batch, batch_idx, "")
