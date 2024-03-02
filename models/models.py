import copy
import math
import logging
import torch
import torchaudio
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from models.content_extractors import (
    WavLMFeatureExtractor,
    HubertFeatureExtractor,
    SpinModelFeatureExtractor
)
from models.speaker_encoders import (
    ByolSpeakerEncoder,
    CoquiSpeakerEncoder,
    DefaultSpeakerEncoder,
    ECAPA2SpeakerEncoder16k,
    RawNet3SpeakerEncoder44k,
)
from models import commons
from models import modules
from models.commons import init_weights, get_padding
from models.so_vits_svc import TextEncoder

def f0_to_coarse(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    # guarantee pitch max and min
    clip_fn = torch.clip if is_torch else np.clip
    f0 = clip_fn(f0, f0_min, f0_max)

    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).int() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 cond_pitch=True,
                 pitch_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.pitch_channels = pitch_channels

        if not cond_pitch:
            pitch_channels = 0

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate,
                              n_layers, gin_channels=self.gin_channels, pitch_channels=pitch_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, pitch=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, pitch=pitch, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, pitch=pitch, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0,
                 cond_f0=False,
                 cond_lang=False,
                 lang_dim=0,
                 num_langs=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        if cond_f0:
            self.f0_emb = nn.Embedding(256, hidden_channels)
        else:
            self.f0_emb = None

        if cond_lang:
            self.lang_emb = nn.Embedding(num_langs, lang_dim)
        else:
            self.lang_emb = None

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=self.gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, f0=None, lang_id=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        if self.f0_emb:
            x = x + self.f0_emb(f0).squeeze(1).transpose(1, 2)
        if self.lang_emb:
            x = x + self.lang_emb(lang_id).unsqueeze(-1) # Use of broadcasting
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1,
                   padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + \
            [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
             for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size,
                            model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(
                mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 c_dim,
                 use_spk_emb,
                 freeze_external_spk,
                 spk_encoder_type,
                 config=None,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.c_dim = c_dim
        self.gin_channels = gin_channels
        self.config = config

        if spk_encoder_type is None and use_spk_emb:
            logging.warning("Speaker encoder model is None and use_spk_emb is True. Speaker embedding will not be used.")
        self.spk_encoder_type = spk_encoder_type
        if self.gin_channels is None and self.spk_encoder_type is None:
            raise ValueError("gin_channels and spk_encoder_model cannot be None at the same time.")
        self.use_spk_emb = use_spk_emb
        self.freeze_external_spk = freeze_external_spk
        if self.use_spk_emb:
            if self.spk_encoder_type == "ByolSpeakerEncoder":
                self.speaker_encoder = ByolSpeakerEncoder(self.config.model.device)
            elif self.spk_encoder_type == "CoquiSpeakerEncoder":
                self.speaker_encoder = CoquiSpeakerEncoder(self.config.model.device)
            elif self.spk_encoder_type == "DefaultSpeakerEncoder":
                self.speaker_encoder = DefaultSpeakerEncoder(self.config.model.device)
            elif self.spk_encoder_type == "ECAPA2SpeakerEncoder16k":
                self.speaker_encoder = ECAPA2SpeakerEncoder16k(self.config.model.device)
            elif self.spk_encoder_type == "RawNet3SpeakerEncoder44k":
                self.speaker_encoder = RawNet3SpeakerEncoder44k(self.config.model.device, self.config.spk_encoder_ckpt)
            else:
                raise ValueError(f"Unknown spk_encoder_model: {self.spk_encoder_type}")
            print(f"Speaker encoder model: {self.spk_encoder_type}")

            if self.gin_channels is None:
                self.gin_channels = self.speaker_encoder.embedding_dim

            if self.freeze_external_spk:
                for param in self.speaker_encoder.parameters():
                    param.requires_grad = False

        self.coarse_f0 = True if "coarse_f0" in self.config.model and self.config.model.coarse_f0 else False
        self.cond_f0_on_flow = True if "cond_f0_on_flow" in self.config.model and self.config.model.cond_f0_on_flow else False
        if not self.cond_f0_on_flow and not self.coarse_f0:
            raise ValueError('You can only uses the f0 conditioning on encoder if it is coarse. Please enable coarse_f0 on config !')

        if self.config.model.content_encoder_type == "wavlm":
            self.c_model = WavLMFeatureExtractor(self.config.model.content_encoder_ckpt, svc_model_sr=self.config.data.sampling_rate)
        elif self.config.model.content_encoder_type == "hubert":
            self.c_model = HubertFeatureExtractor(self.config.model.content_encoder_ckpt, svc_model_sr=self.config.data.sampling_rate)
        elif self.config.model.content_encoder_type == "spin":
            self.c_model = SpinModelFeatureExtractor(
                self.config.model.content_encoder_config,
                self.config.model.content_encoder_ckpt,
                svc_model_sr=self.config.data.sampling_rate
            )
        elif self.config.model.content_encoder_type == None:
            self.c_model = None
        else:
            raise ValueError(f"Unknown content_encoder_type: {self.config.model.content_encoder_type}")

        if self.config.model.post_content_encoder_type == "freevc-bottleneck":
            self.enc_p = Encoder(
                c_dim,
                inter_channels,
                hidden_channels,
                5,
                1,
                16,
                cond_f0=not self.cond_f0_on_flow,
                cond_lang=self.config.data.get("use_lang_emb", False),
                num_langs=self.config.data.get("num_langs", 7),
                lang_dim=self.config.data.get("lang_dim", 192),
            )
        elif self.config.model.post_content_encoder_type == "vits-encoder-with-uv-emb":
            # transformer encoder with voice/unvoice embedding and pitch embedding
            self.enc_p = TextEncoder(
                c_dim,
                inter_channels,
                hidden_channels,
                filter_channels=filter_channels,
                n_heads=n_heads,
                n_layers=n_layers,
                kernel_size=kernel_size,
                p_dropout=p_dropout,
                cond_f0=not self.cond_f0_on_flow,
                cond_lang=self.config.data.get("use_lang_emb", False),
                num_langs=self.config.data.get("num_langs", 7),
                lang_dim=self.config.data.get("lang_dim", 192),
            )
        else:
            raise ValueError(f"Unknown post_content_encoder_type: {self.config.post_content_encoder_type}")

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                             upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=self.gin_channels)
        self.enc_q = Encoder(spec_channels, inter_channels,
                             hidden_channels, 5, 1, 16, gin_channels=self.gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=self.gin_channels, cond_pitch=self.cond_f0_on_flow, pitch_channels=1)

    def get_spk_emb(self, y=None, mel=None):
        if self.spk_encoder_type == "DefaultSpeakerEncoder":
            assert mel is not None, "mel is None"
            g = self.speaker_encoder(mel.transpose(1, 2))
            g = g.unsqueeze(-1)
        elif self.spk_encoder_type == "ByolSpeakerEncoder":
            assert y is not None, "y is None"
            g = self.speaker_encoder(y)
            g = g.unsqueeze(-1)
        elif self.spk_encoder_type == "CoquiSpeakerEncoder":
            if self.sampling_rate != self.speaker_encoder.audio_config["sample_rate"]:
                y_spk = torchaudio.functional.resample(
                        y,
                        orig_freq=self.hps.data.sampling_rate,
                        new_freq=self.speaker_encoder.audio_config["sample_rate"],
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method="kaiser_window",
                        beta=14.769656459379492,
                )
            else:
                y_spk = y
            g = self.speaker_encoder.forward(y_spk.contiguous(), l2_norm=True).unsqueeze(-1)
        elif self.spk_encoder_type == "RawNet3":
            g = self.speaker_encoder(y)
            g = g.unsqueeze(-1)
        elif self.spk_encoder_type == "ECAPA2SpeakerEncoder16k":
            g = self.speaker_encoder(y)
            g = g.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown spk_encoder_model: {self.spk_encoder_type}")

        return g

    def forward(self, spec, y=None, c=None, g=None, mel=None, c_lengths=None, spec_lengths=None, pitch=None, lang_id=None):

        if c is None:
            if self.c_model is None:
                raise ValueError("c is None and c_model is also None")
            if y is None:
                raise ValueError("c is None and y is also None")
            c = self.c_model.extract_features(y)

        # c is smaller than spec so interpolate c to the size of spec on dim 1
        if c.size(2) != mel.size(2):
            c = torch.nn.functional.interpolate(
                    c.unsqueeze(1), size=[c.size(1), mel.size(2)], mode="nearest").squeeze(1)
            # reset c lenghts to the new lenghts
            c_lengths = spec_lengths

        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths == None:
            spec_lengths = (torch.ones(spec.size(0)) *
                            spec.size(-1)).to(spec.device)

        if g is None:
            g = self.get_spk_emb(y, mel)
        assert g is not None, "g is None. Check configuration of speaker encoder model or g input on forward method"

        # ToDo: Implement denormalizator of pitch (F0Decoder) on https://github.com/svc-develop-team/so-vits-svc/blob/58865936d6b3e6dbca55ef2c7013bea62253431a/models.py#L369
        if self.coarse_f0:
            pitch = f0_to_coarse(pitch).detach()

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths, f0=pitch if not self.cond_f0_on_flow else None, lang_id=lang_id)

        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g, pitch=pitch.float() if self.cond_f0_on_flow else None)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c=None, y=None, g=None, mel=None, c_lengths=None, pitch=None, lang_id=None):

        if c is None:
            if self.c_model is None:
                raise ValueError("c is None and c_model is also None")
            if y is None:
                raise ValueError("c is None and y is also None")
            with torch.no_grad():
                c = self.c_model.extract_features(y)
        # c is smaller than pitch so pad c to the size of pitch on dim 2 (time dimention). Uses pitch on inference because mel spec is from target speaker not source
        if c.size(2) != pitch.size(2):
            c = torch.nn.functional.interpolate(
                    c.unsqueeze(1), size=[c.size(1), pitch.size(2)], mode="nearest").squeeze(1)
            # reset c_lenghts
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if g is None:
            g = self.get_spk_emb(y, mel)
        assert g is not None, "g is None. Check configuration of speaker encoder model or g input on forward method"

        if self.coarse_f0:
            pitch = f0_to_coarse(pitch).detach()

        z_p, _, _, c_mask = self.enc_p(c, c_lengths, f0=pitch if not self.cond_f0_on_flow else None, lang_id=lang_id)
        z = self.flow(z_p, c_mask, g=g, pitch=pitch.float() if self.cond_f0_on_flow else None, reverse=True)
        o = self.dec(z * c_mask, g=g)

        return o

    def voice_conversion(self, c_src=None, y_src=None, y_tgt=None, g_tgt=None, mel_tgt=None, c_lengths=None, pitch_tgt=None, lang_id_src=None):
        if c_src is None:
            if self.c_model is None:
                raise ValueError("c_src is None and c_model is also None")
            if y_src is None:
                raise ValueError("c_src is None and y_src is also None")
            with torch.no_grad():
                c_src = self.c_model.extract_features(y_src)
        # c is smaller than pitch so pad c to the size of pitch on dim 2 (time dimention). Uses pitch on inference because mel spec is from target speaker not source
        if c_src.size(2) != pitch_tgt.size(2):
            c_src = torch.nn.functional.interpolate(
                    c_src.unsqueeze(1), size=[c_src.size(1), pitch_tgt.size(2)], mode="nearest").squeeze(1)
            # reset c_lenghts
            c_lengths = (torch.ones(c_src.size(0)) * c_src.size(-1)).to(c_src.device)

        if c_lengths == None:
            c_lengths = (torch.ones(c_src.size(0)) * c_src.size(-1)).to(c_src.device)

        if g_tgt is None:
            g_tgt = self.get_spk_emb(y_tgt, mel_tgt)
        assert g_tgt is not None, "g is None. Check configuration of speaker encoder model or g input on forward method"

        if self.coarse_f0:
            pitch_tgt = f0_to_coarse(pitch_tgt).detach()

        z_p, _, _, c_mask = self.enc_p(c_src, c_lengths, f0=pitch_tgt if not self.cond_f0_on_flow else None, lang_id=lang_id_src)
        z = self.flow(z_p, c_mask, g=g_tgt, pitch=pitch_tgt.float() if self.cond_f0_on_flow else None, reverse=True)
        o = self.dec(z * c_mask, g=g_tgt)

        return o
