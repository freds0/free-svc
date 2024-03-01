import torch
import torchaudio
import librosa
import logging

from torch import nn

from .speaker_encoder.audio import wav_to_mel_spectrogram
from .speaker_encoder.audio import preprocess_wav as speaker_encoder_preprocess
from .speaker_encoder.voice_encoder import SpeakerEncoder

from .ssl_singer_identity.singer_identity import load_model as load_ssl_singer_identity_model

try:
    from TTS.tts.utils.speakers import SpeakerManager
except ImportError:
    logging.warning("TTS is not installed, COQUI speaker encoder will not be available")

from .clova.models.RawNet3 import MainModel as RawNet3
from .clova.SpeakerNet import SpeakerNet

from huggingface_hub import hf_hub_download


class ECAPA2SpeakerEncoder16k(nn.Module):

    def __init__(self, device, spk_encoder_ckpt='Jenthe/ECAPA2', half=False):
        super().__init__()
        self.device = device
        model_file = hf_hub_download(repo_id=spk_encoder_ckpt, filename='ecapa2.pt', cache_dir=None)
        self.speaker_encoder = torch.jit.load(model_file, map_location=device)
        if device == "cuda" and half:
            self.speaker_encoder.half()

    def get_speaker_embedding(self, filepath):
        audio, _ = librosa.load(filepath, sr=16000, mono=True)
        with torch.jit.optimized_execution(False):
            return self.speaker_encoder(torch.from_numpy(audio).to(self.device).unsqueeze(0)).detach().cpu()

    @property
    def model(self):
        return self.speaker_encoder

    @property
    def embedding_dim(self):
        return 192

    def forward(self, x):
        return self.speaker_encoder(x)

class ByolSpeakerEncoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.speaker_encoder = load_ssl_singer_identity_model("byol").to(device)

    def get_speaker_embedding(self, filepath):
        return self.speaker_encoder(filepath)

    @property
    def model(self):
        return self.speaker_encoder

    @property
    def embedding_dim(self):
        return 1000

    def forward(self, x):
        return self.speaker_encoder(x)

class RawNet3SpeakerEncoder44k(nn.Module):

    def __init__(self, device, spk_encoder_ckpt):
        super().__init__()
        self.device = device
        self.speaker_encoder = RawNet3()
        self.speaker_encoder = SpeakerNet(RawNet3, nOut=256, encoder_type="ECA", sinc_stride=10)  # TODO: make it configurable
        self.speaker_encoder.loadParameters(spk_encoder_ckpt)
        self.speaker_encoder.eval()
        self.speaker_encoder.to(device)

    def get_speaker_embedding(self, filepath):
        audio, _ = librosa.load(filepath, sr=44100, mono=True)
        return self.speaker_encoder(torch.from_numpy(audio).to(self.device)).detach().cpu()

    @property
    def model(self):
        return self.speaker_encoder

    @property
    def embedding_dim(self):
        return 256

    def forward(self, x):
        return self.speaker_encoder(x)

class CoquiSpeakerEncoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.speaker_manager = SpeakerManager()
        self.speaker_manager.load()
        self.speaker_manager.to(self.device)

    def get_speaker_embedding(self, filepath):
        waveform, sampling_rate = torchaudio.load(filepath)
        if sampling_rate != self.speaker_encoder.audio_config["sample_rate"]:
            spk_waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=sampling_rate,
                    new_freq=self.speaker_encoder.audio_config["sample_rate"],
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="kaiser_window",
                    beta=14.769656459379492,
            )
        else:
            spk_waveform = waveform

        return self.speaker_encoder.forward(spk_waveform.contiguous(), l2_norm=True).unsqueeze(-1)

    @property
    def model(self):
        return self.speaker_encoder

    @property
    def embedding_dim(self):
        return 512

    def forward(self, x):
        return self.speaker_encoder(x)


class DefaultSpeakerEncoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.speaker_encoder = SpeakerEncoder(device=device)

    def get_speaker_embedding(self, filepath):
        return self.speaker_encoder.get_embedding(filepath)

    @property
    def model(self):
        return self.speaker_encoder

    @property
    def embedding_dim(self):
        return 256

    def forward(self, x):
        return self.speaker_encoder(x)