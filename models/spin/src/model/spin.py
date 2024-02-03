import copy
import logging
from collections import defaultdict
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from models.spin.src.data import (
    AudioPretrainDataset,
    MaxLengthBatchSampler,
    MaxLengthDistributedSampler,
    collate_fn,
)
from models.spin.src.nn import DNN, HuBERT, SwavVQDisentangle, WavLM
from models.spin.src.util import compute_show_pnmi, get_scheduler, update_padding_mask

from .base import BaseModel

logger = logging.getLogger("spin")


def get_pred_head(type_name: str, hid_dim: int, config: dict) -> Union[None, nn.Module]:
    if type_name == "None":
        return None
    if type_name == "DNN":
        return DNN(hid_dim, **config)
    raise NotImplementedError(type_name)


def get_loss(type_name: str, hid_dim: int, config: dict) -> nn.Module:
    if type_name == "SwavVQDisentangle":
        return SwavVQDisentangle(hid_dim, **config)
    raise NotImplementedError(type_name)


class SpinModel(BaseModel):
    def __init__(self, config, num_view: int = 2) -> None:
        super().__init__(config)
        
        config = copy.deepcopy(config)
        config = config["model"]
        logger.info(f"Model config: {config}")

        self.encoder_type = config["encoder"].pop("type", "HuBERT")
        self.pred_head_type = config["pred_head"].pop("type", "DNN")
        self.loss_type = config["loss"].pop("type", "SwavVQDisentangle")

        logger.info(f"Encoder:          {self.encoder_type}")
        logger.info(f"Prediction head:  {self.pred_head_type}")
        logger.info(f"Loss:             {self.loss_type}")

        # Setup number of views
        self.normalize = config.get("normalize", False)
        self.num_view = num_view
        assert num_view == 2, num_view  # NOTE: currently we support 2 views only

        # Setup encoder model
        if self.encoder_type in {"HuBERT", "WavLM"}:
            self.use_layer = config["encoder"].pop("use_layer", 12)
            self.encoder = eval(self.encoder_type)(**config["encoder"])
            hid_dim = self.encoder.hidden_sizes[self.use_layer]
            self.encoder_rate = 320
            logger.info(f"Taking features from layer {self.use_layer}")
        else:
            raise NotImplementedError(self.encoder_type)

        # All layers to be processed
        self.use_layers = [self.use_layer]
        logger.info(f"All selected layers: {self.use_layers}")

        # Setup prediction head
        if len(self.use_layers) == 1:
            self.pred_head = get_pred_head(
                self.pred_head_type, hid_dim, config["pred_head"]
            )
            hid_dim = self.pred_head.out_dim
        else:
            self.pred_head = nn.ModuleList(
                [
                    get_pred_head(self.pred_head_type, hid_dim, config["pred_head"])
                    for _ in self.use_layers
                ]
            )
            hid_dim = self.pred_head[0].out_dim

        # Setup loss function
        self.loss_module = get_loss(self.loss_type, hid_dim, config["loss"])

        # Validation
        self.val_uid2hyp = {}

    def normalize_wavs(
        self,
        wavs: torch.Tensor,
        wavs_len: torch.LongTensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            for i in len(wavs):
                wavs[i, : wavs_len[i]] = F.layer_norm(
                    wavs[i, : wavs_len[i]], wavs_len[i]
                )
        return wavs

    def forward_features(
        self,
        wavs: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ):
        if self.encoder_type in {"HuBERT", "WavLM"}:
            feat_list, feat_len, padding_mask = self.encoder(wavs, padding_mask)
            repr_list = [feat_list[l] for l in self.use_layers]

        return repr_list, feat_list, feat_len, padding_mask

    def forward_pred_head(
        self, feat: torch.Tensor, feat_len: torch.LongTensor, i: int = None
    ):
        if len(self.use_layers) == 1:
            return self.pred_head(feat, feat_len)
        else:
            assert isinstance(i, int), i
            return self.pred_head[i](feat, feat_len)

    def forward(self, batch, feat_only: bool = False):
        # NOTE: padding_mask is 1 when the position is padded
        wavs, wavs_len, padding_mask = batch

        wavs.masked_fill_(padding_mask, 0.0)

        # Normalize wavs
        if self.normalize:
            wavs = self.normalize_wavs(wavs, wavs_len)

        # Extract features
        repr_list, feat_list, feat_len, padding_mask = self.forward_features(
            wavs, padding_mask
        )
        padding_mask = update_padding_mask(padding_mask, repr_list[0].shape[1])

        # Prediction head
        repr_list = [
            self.forward_pred_head(repr, feat_len, i)
            for i, repr in enumerate(repr_list)
        ]

        # Return results
        if feat_only:
            outputs = {
                "repr_list": repr_list,
                "feat_len": feat_len,
                "feat_list": feat_list,
                "padding_mask": padding_mask,
            }
            if self.loss_type == "SwavVQDisentangle":
                if self.loss_module.l2_norm:
                    outputs["repr_list"] = F.normalize(repr_list[0], dim=-1)
                logits, codes = self.loss_module.produce_targets(
                    outputs["repr_list"], normalized=True
                )
                outputs["logits"] = logits
                outputs["codes"] = codes

            return outputs

        # feat: (Batch * View, Time, Dim)
        # Split batch into views
        repr_views = [
            [
                r[i :: self.num_view][~padding_mask[i :: self.num_view]]
                for r in repr_list
            ]
            for i in range(self.num_view)
        ]

        # Computes loss from each pair of views
        total_loss = 0
        loss_res = defaultdict(list)

        # Main loss
        if self.loss_type in {"SwavVQDisentangle"}:
            res = self.loss_module.cal_loss(repr_views[0][0], repr_views[1][0])
        total_loss += res.pop("loss")
        for k in res:
            loss_res[f"{k}"].append(res[k])

        for k in loss_res:
            loss_res[k] = sum(loss_res[k]) / len(loss_res[k])

        return total_loss, loss_res

    def training_step(self, batch, batch_idx):
        total_loss, loss_res = self.forward(batch)

        self.log("loss", total_loss)
        for k, v in loss_res.items():
            if k == "acc":
                self.log(k, v, prog_bar=True)
            else:
                self.log(k, v)

        return total_loss

    def validation_step(self, batch, batch_idx):
        wav_list, wav_len, padding_mask, uid_list = batch
        results = self.forward(
            (wav_list, wav_len, padding_mask),
            feat_only=True,
        )
        codes = results.get("codes", None)
        if codes is None:
            return

        code = codes.cpu().numpy()
        feat_len = results["feat_len"]
        for i, (uid, c) in enumerate(zip(uid_list, code)):
            self.val_uid2hyp[uid] = c[: feat_len[i]]

    def on_validation_epoch_end(self) -> None:
        check_1, check_2 = True, True
        try:
            uid2ref = self.trainer.val_dataloaders[0].dataset.uid2refs
        except:
            check_1 = False

        if not check_1:
            try:
                uid2ref = self.trainer.val_dataloaders.dataset.uid2refs
            except:
                check_2 = False

        if not check_1 and not check_2:
            logger.info("Cannot find uid2ref in validation dataloader, skip PNMI")
            return

        if len(self.val_uid2hyp) == 0:
            return

        res = compute_show_pnmi(
            uid2ref, self.val_uid2hyp, upsample=self.encoder_rate // 160
        )
        self.log("cls_pur", res["cls_pur"])
        self.log("phn_pur", res["phn_pur"])
        self.log("pnmi", res["pnmi"])
        self.val_uid2hyp.clear()

    def on_before_zero_grad(self, optimizer) -> None:
        if self.loss_type == "SwavVQDisentangle":
            self.loss_module.normalize_codebook()

        return super().on_before_zero_grad(optimizer)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        try:
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
            logger.info(f"Update epoch for batch sampler to {self.current_epoch}")
        except:
            logger.warn(
                "Unable to update epoch for batch sampler (possibly using fixed batch_size)"
            )

    def configure_optimizers(self):
        params = []
        if self.encoder_type in {"HuBERT", "WavLM"}:
            params += self.encoder.trainable_parameters()
        else:
            params += list(self.encoder.parameters())

        if self.pred_head:
            params += list(self.pred_head.parameters())
        params += list(self.loss_module.parameters())

        optimizer = getattr(torch.optim, self.config["optim"]["optimizer"]["name"])(
            params, **self.config["optim"]["optimizer"]["args"]
        )

        if self.config["optim"].get("scheduler", None):
            scheduler = get_scheduler(
                self.config["optim"]["scheduler"]["name"],
                optimizer,
                **self.config["optim"]["scheduler"]["args"],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def set_random_seed(self, seed: int = 7122):
        self.seed = seed

    def set_njobs(self, njobs: int = 0):
        self.njobs = njobs

    def set_use_ddp(self, use_ddp: bool = False):
        self.use_ddp = use_ddp

    def train_dataloader(self):
        dataset = AudioPretrainDataset(**self.config["data"])
        if "batch_len" in self.config["hparam"]:
            if self.use_ddp:
                sampler = MaxLengthDistributedSampler(
                    dataset,
                    dataset.data_lens,
                    max_length=self.config["hparam"]["batch_len"],
                    cropped_length=self.config["data"]["random_crop_len"],
                    shuffle=True,
                    drop_last=True,
                    seed=self.seed,
                )
            else:
                sampler = MaxLengthBatchSampler(
                    dataset.data_lens,
                    max_length=self.config["hparam"]["batch_len"],
                    cropped_length=self.config["data"]["random_crop_len"],
                    shuffle=True,
                    drop_last=True,
                    seed=self.seed,
                )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.njobs,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        elif "batch_size" in self.config["hparam"]:
            loader = DataLoader(
                dataset,
                batch_size=self.config["hparam"]["batch_size"],
                num_workers=self.njobs,
                pin_memory=True,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True,
            )

        return loader
