import logging
from typing import List

import torch
import torch.nn.functional as F
from s3prl.upstream.wavlm.modules import GradMultiply
from s3prl.upstream.wavlm.WavLM import WavLM as WavLMModel
from s3prl.upstream.wavlm.WavLM import WavLMConfig
from s3prl.util.download import _urls_to_filepaths
from torch import nn

from models.spin.src.util import (
    freeze_module,
    init_module_bert,
    init_module_cnn,
    init_module_pos_conv,
    padding_to_len,
)

logger = logging.getLogger("wavlm")


class WavLM(nn.Module):
    def __init__(
        self,
        path_or_url: str = None,
        refresh: bool = False,
        pre_normalize: bool = False,
        normalize: bool = False,
        feat_select: str = "x",
        randomize_all: bool = False,
        randomize_layers: List[int] = [],
        freeze_all: bool = False,
        freeze_layers: List[int] = [],
    ):
        super().__init__()

        ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base.pt"
        if path_or_url is not None:
            ckpt = path_or_url
        if ckpt.startswith("https"):
            ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

        checkpoint = torch.load(ckpt)
        self.cfg = WavLMConfig(checkpoint["cfg"])
        self.model = WavLMModel(self.cfg)
        self.model.load_state_dict(checkpoint["model"])

        self.wav_normalize = self.cfg.normalize
        self.pre_normalize = pre_normalize
        self.normalize = normalize
        self.num_layers = self.cfg.encoder_layers + 1  # CNN + 12 Transformer
        self.hidden_sizes = [self.cfg.encoder_embed_dim] * self.num_layers

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        self.feat_select = 0
        if feat_select == "att":
            self.feat_select = 1

        logger.info(f"Feature selection: {feat_select}")
        logger.info(
            f"Randomize all = {randomize_all} (randomize layers = {randomize_layers})"
        )
        logger.info(f"Freeze all = {freeze_all} (freeze layers = {freeze_layers})")

        self.randomize_all = randomize_all
        self.randomize_layers = randomize_layers
        self.freeze_all = freeze_all
        self.freeze_layers = freeze_layers

        self.freeze_cnn = (0 in freeze_layers) or self.freeze_all
        self.freeze_pos = ("pos" in freeze_layers) or self.freeze_all

        # Randomize weights
        if randomize_all:
            randomize_layers = ["pos"] + list(range(self.num_layers))
        if len(randomize_layers) > 0:
            for i in randomize_layers:
                if i == 0:
                    self.model.feature_extractor.apply(init_module_cnn)
                    self.model.layer_norm.reset_parameters()
                    if self.model.post_extract_proj is not None:
                        self.model.post_extract_proj.reset_parameters()
                elif i == "pos":
                    self.model.encoder.pos_conv.apply(init_module_pos_conv)
                else:
                    self.model.encoder.layers[i - 1].apply(init_module_bert)
                    if i == self.num_layers - 1 and self.model.encoder.layer_norm_first:
                        self.model.encoder.layer_norm.reset_parameters()

        # Freeze weights
        if freeze_all:
            freeze_module(self.model)
        elif len(freeze_layers) > 0:
            for i in freeze_layers:
                if i == 0:
                    self.model.feature_grad_mult = 0.0
                    freeze_module(self.model.feature_extractor)
                    freeze_module(self.model.layer_norm)
                    if self.model.post_extract_proj is not None:
                        freeze_module(self.model.post_extract_proj)
                elif i == "pos":
                    freeze_module(self.model.encoder.pos_conv)
                else:
                    assert isinstance(i, int), i
                    freeze_module(self.model.encoder.layers[i - 1])

        if not self.freeze_cnn:
            self.model.feature_grad_mult = 1.0

    def trainable_parameters(self):
        params = []

        if self.freeze_all:
            return []

        if not self.freeze_all and len(self.freeze_layers) == 0:
            logger.info("Trains the entire model")
            return self.model.parameters()

        params = []
        for i in ["pos"] + list(range(self.num_layers)):
            if i in self.freeze_layers:
                continue
            if i == 0:
                params += list(self.model.feature_extractor.parameters())
                params += list(self.model.layer_norm.parameters())
                if self.model.post_extract_proj is not None:
                    params += list(self.model.post_extract_proj.parameters())
            elif i == "pos":
                params += list(self.model.encoder.pos_conv.parameters())
            else:
                params += list(self.model.encoder.layers[i - 1].parameters())
                if i == self.num_layers - 1 and self.model.encoder.layer_norm_first:
                    params += list(self.model.encoder.layer_norm.parameters())

        return params

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.feature_grad_mult > 0:
            features = self.model.feature_extractor(x)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.model.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.model.feature_extractor(x)

        return features

    def forward(
        self,
        wavs: torch.FloatTensor,
        padding_mask: torch.BoolTensor,
    ):
        if self.pre_normalize:
            with torch.no_grad():
                wavs = (wavs - wavs[~padding_mask].mean()) / (
                    wavs[~padding_mask].std() + 1e-5
                )

        if self.wav_normalize:
            with torch.no_grad():
                wav_len = padding_to_len(padding_mask)
                for i in range(len(wavs)):
                    wavs[i, : wav_len[i]] = F.layer_norm(
                        wavs[i, : wav_len[i]], (wav_len[i],)
                    )

        features = self.forward_features(wavs).transpose(1, 2)
        if self.freeze_cnn:
            with torch.no_grad():
                features = self.model.layer_norm(features)
                if self.model.post_extract_proj is not None:
                    features = self.model.post_extract_proj(features)
        else:
            features = self.model.layer_norm(features)
            if self.model.post_extract_proj is not None:
                features = self.model.post_extract_proj(features)

        if padding_mask is not None:
            padding_mask = self.model.forward_padding_mask(features, padding_mask)

        x = self.model.dropout_input(features)

        # feature: (B, T, D), float
        # x: (B, T, D), float
        # padding_mask: (B, T), bool

        if self.freeze_all:
            with torch.no_grad():
                _, layer_results = self.model.encoder(
                    x,
                    padding_mask=padding_mask,
                )
        else:
            _, layer_results = self.model.encoder(
                x,
                padding_mask=padding_mask,
                freeze_pos=self.freeze_pos,
                freeze_layers=self.freeze_layers,
            )

        feat_list = [features] + [
            feat[self.feat_select].transpose(0, 1) for feat in layer_results
        ]

        if self.normalize:
            feat_list = [F.layer_norm(feat, feat.shape[-1:]) for feat in feat_list]

        feat_len = padding_to_len(padding_mask)

        return feat_list, feat_len, padding_mask
