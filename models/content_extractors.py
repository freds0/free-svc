import torch
from torch import nn

import utils

import yaml
import torchaudio
from mel_processing import mel_processing
from models.wavlm import WavLM, WavLMConfig
from models.spin import SpinModel, spin_collate_fn

import logging
logging.getLogger('numba').setLevel(logging.WARNING)
from transformers import HubertModel


class SpinModelFeatureExtractor(nn.Module):
    def __init__(self, config, checkpoint_path, freeze=True, svc_model_sr=16000):
        super().__init__()
        if isinstance(config, str) and config.split(".")[-1] in {"yaml", "yml"}:
            config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        self.model = SpinModel(config)
        print(self.model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.extractor_sr = 16000
        self.svc_model_sr = svc_model_sr
        self.freeze = freeze
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def extract_features(self, y, layer=-1):
        if self.svc_model_sr != self.extractor_sr:
            y = torchaudio.functional.resample(
                y,
                orig_freq=self.svc_model_sr,
                new_freq=self.extractor_sr,
            )

        wav_list, wav_len, padding_mask = spin_collate_fn(y)
        wav_list.cuda()
        padding_mask.cuda()
        batch = wav_list, wav_len, padding_mask

        if self.freeze:
            with torch.no_grad():
                c = self.model.forward_features(wavs=wav_list, padding_mask=padding_mask)[1][layer]
        else:
            c = self.model.forward_features(wavs=wav_list, padding_mask=padding_mask)[1][layer]

        c = c.transpose(1, 2)
        return c

class WavLMFeatureExtractor(WavLM):
    def __init__(self, checkpoint_path, freeze=True, svc_model_sr=16000):
        checkpoint = torch.load(checkpoint_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM.__init__(self, cfg)
        self.cuda()
        self.load_state_dict(checkpoint['model'])
        self.extractor_sr = 16000
        self.svc_model_sr = svc_model_sr
        self.freeze = freeze
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def extract_features(self, y, **kwargs):
        y = y.squeeze()
        if y.ndim == 1:
            y = y.unsqueeze(0)

        if self.svc_model_sr != self.extractor_sr:
            y = torchaudio.functional.resample(
                y,
                orig_freq=self.svc_model_sr,
                new_freq=self.extractor_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )

        if self.freeze:
            with torch.no_grad():
                c = super().extract_features(y, **kwargs)[0]
        else:
            c = super().extract_features(y, **kwargs)[0]

        c = c.transpose(1, 2)

        # This is necessary because the features in the time dimension has one element
        # less than other features:
        c_padded = torch.zeros((c.shape[0], c.shape[1], c.shape[2]+1), device=c.device)
        c_padded[:, :, :-1] = c
        return c_padded

class HubertFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path, freeze=True, svc_model_sr=16000):
        super().__init__()
        self.model = HubertModel.from_pretrained(checkpoint_path)
        self.extractor_sr = 16000
        self.svc_model_sr = svc_model_sr
        self.freeze = freeze
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def extract_features(self, y, **kwargs):
        y = y.squeeze()
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if self.svc_model_sr != self.extractor_sr:
            y = torchaudio.functional.resample(
                y,
                orig_freq=self.svc_model_sr,
                new_freq=self.extractor_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        if self.freeze:
            with torch.no_grad():
                feats = self.model(y)["last_hidden_state"]
        else:
            feats = self.model(y)["last_hidden_state"]
        return feats.transpose(1, 2)
