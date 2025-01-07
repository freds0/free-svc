import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import sosfilt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from models.spin.src.util import len_to_padding

from .audio import change_gender, change_gender_f0, params2sos

logger = logging.getLogger("dataset")

Qmin, Qmax = 2, 5


def read_phn(tsv_path, rm_stress=True):
    uid2phns = {}
    with open(tsv_path) as f:
        for line in f:
            uid, phns = line.rstrip().split("\t")
            phns = phns.split(",")
            if rm_stress:
                phns = [re.sub("[0-9]", "", phn) for phn in phns]
            uid2phns[uid] = phns
    return uid2phns


class AudioPretrainDataset(Dataset):
    def __init__(
        self,
        json_dir: str,
        splits: List[str],
        spk2info: str = None,
        min_audio_len: int = 1,
        max_audio_len: int = 1_000_000,
        sample_rate: int = 16_000,
        random_crop_len: int = -1,
        use_ratio: float = 1.0,
        normalize: bool = False,
        nansy_both: bool = False,
    ) -> None:
        super().__init__()

        logger.info(f"Loading audio from: {json_dir}")
        logger.info(f"Splits : {splits}")
        self.split_type = "train" if splits[0].startswith("train") else "valid"
        self.return_augmented = True

        # Data augmentation config
        self.random_crop_len = random_crop_len
        self.random_crop_len_lab = random_crop_len // 320
        self.use_ratio = use_ratio
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.nansy_aug = spk2info is not None

        if self.nansy_aug:
            # Load speaker information
            logger.info("Apply NANSY speaker perturbation")
            # ref: https://arxiv.org/abs/2110.14513
            self.num_view = 2
            self.nansy_both = nansy_both

            with open(spk2info, "rb") as fp:
                self.spk2info = pickle.load(fp)
                self.spk2info = self.spk2info[self.split_type]

            self.rng = np.random.default_rng()
            self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))

        self.num_view = 2 if self.nansy_aug else 1

        # Load from .json files
        data_list = []
        for s in splits:
            with open(os.path.join(json_dir, s + ".json"), "r") as fp:
                data_list += json.load(fp)

        # Preserve certain ratio of data
        if self.use_ratio < 1.0:
            logger.info(
                f"Using only {self.use_ratio * 100:.0f}% of randomly chosen data"
            )
            random.shuffle(data_list)
            data_list = data_list[: int(len(data_list) * self.use_ratio)]

        # Remove files that are too long or too short
        logger.info(
            f"Removing files shorter than {min_audio_len} or longer than {max_audio_len} frames"
        )
        orig_tot_len = sum([l for _, l in data_list]) / sample_rate / 3600.0
        orig_num_file = len(data_list)
        data_list = [
            (p, l) for p, l in data_list if l >= min_audio_len and l <= max_audio_len
        ]
        new_tot_len = sum([l for _, l in data_list]) / sample_rate / 3600.0
        logger.info(f"Original audio files: {orig_num_file} ({orig_tot_len:.2f} hrs)")
        logger.info(f"Final audio files:    {len(data_list)} ({new_tot_len:.2f} hrs)")

        # Sort by length (long to short)
        data_list = sorted(data_list, key=lambda x: x[1], reverse=True)

        # Extract speaker
        spk_list = [Path(p).stem.split("-")[0] for p, _ in data_list]
        name_list = [Path(p).stem for p, _ in data_list]
        self.data = [
            (p, l, s, n) for (p, l), s, n in zip(data_list, spk_list, name_list)
        ]
        self.data_lens = [l for _, l, _, _ in self.data]

        if self.nansy_aug:
            # Check speaker
            data_spk_set = set(spk_list)
            avail_spk_set = set(self.spk2info.keys())
            assert all(s in avail_spk_set for s in data_spk_set), "Missing speakers!"
            logger.info(f"Total {len(data_spk_set)} speakers")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[torch.FloatTensor]:
        # each sample: (audio file path, number of samples, speaker id)
        path, num_frames, spk, uid = self.data[index]

        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)

        wav = wav.astype(np.float32)

        if not self.return_augmented:
            return torch.from_numpy(wav)

        # Randomly crop wave length
        if self.random_crop_len > 0 and len(wav) > self.random_crop_len:
            idx = np.random.randint(0, len(wav) - self.random_crop_len)
            wav = wav[idx : idx + self.random_crop_len]

        # Apply NANSY speaker perturbation
        if self.nansy_aug:
            if self.nansy_both:
                wavs = [
                    self.perturb_speaker(path, wav, sr, spk)
                    for _ in range(self.num_view)
                ]
            else:
                wavs = [wav] + [
                    self.perturb_speaker(path, wav, sr, spk)
                    for _ in range(self.num_view - 1)
                ]
        else:
            wavs = [wav for _ in range(self.num_view)]

        wavs = [torch.FloatTensor(w) for w in wavs]
        if self.normalize:
            wavs = [F.layer_norm(w, w.shape) for w in wavs]

        return wavs

    def get_spk_info(self, spk: str):
        _, (lo, hi, _) = self.spk2info[spk]
        if lo == 50:
            lo = 75
        if spk == "1447":
            lo, hi = 60, 400
        return lo, hi

    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin) ** z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav

    def random_formant_f0(self, wav, sr, spk):
        lo, hi = self.get_spk_info(spk)

        ratio_fs = self.rng.uniform(1, 1.4)
        coin = self.rng.random() > 0.5
        ratio_fs = coin * ratio_fs + (1 - coin) * (1 / ratio_fs)

        ratio_ps = self.rng.uniform(1, 2)
        coin = self.rng.random() > 0.5
        ratio_ps = coin * ratio_ps + (1 - coin) * (1 / ratio_ps)

        ratio_pr = self.rng.uniform(1, 1.5)
        coin = self.rng.random() > 0.5
        ratio_pr = coin * ratio_pr + (1 - coin) * (1 / ratio_pr)

        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)

        return ss

    def fixed_formant_f0(self, wav, sr, spk):
        _, (lo, hi, _) = self.spk2info[spk]

        if lo == 50:
            lo = 75
            ratio_fs, f0_med, ratio_pr = 1.2, 300, 1.2
        else:
            ratio_fs, f0_med, ratio_pr = 0.8, 100, 0.8

        ss = change_gender_f0(wav, sr, lo, hi, ratio_fs, f0_med, ratio_pr)
        return ss

    def perturb_speaker(self, path, wav, sr, spk):
        if self.split_type == "train":
            # Speaker perturbation
            try:
                wav_p = self.random_formant_f0(wav, sr, spk)
            except UserWarning:
                wav_p = np.copy(wav)
                logger.info(f"Praat warning - {path}")
            except RuntimeError:
                wav_p = np.copy(wav)
                logger.info(f"Praat Error - {path}")
            wav_p = self.random_eq(wav_p, sr)
        else:
            try:
                wav_p = self.fixed_formant_f0(wav, sr, spk)
            except UserWarning:
                wav_p = np.copy(wav)
                logger.info(f"Praat warning - {path}")
            except RuntimeError:
                wav_p = np.copy(wav)
                logger.info(f"Praat Error - {path}")

        wav_p = np.clip(wav_p, -1.0, 1.0)
        return wav_p


class AudioPretrainPnmiValDataset(Dataset):
    def __init__(
        self,
        json_dir: str,
        phn_dir: str,
        splits: List[str],
        min_audio_len: int = 1,
        max_audio_len: int = 1_000_000,
        sample_rate: int = 16_000,
        crop_len: int = 160_000,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        logger.info(f"Loading audio from: {json_dir}")
        logger.info(f"Loading phoneme alignments from: {phn_dir}")
        logger.info(f"Splits : {splits}")
        self.split_type = "train" if splits[0].startswith("train") else "valid"

        # Data augmentation config
        self.crop_len = crop_len
        self.normalize = normalize
        self.sample_rate = sample_rate

        # Load from .json files
        data_list = []
        for s in splits:
            with open(os.path.join(json_dir, s + ".json"), "r") as fp:
                data_list += json.load(fp)

        # Load from .tsv files
        self.uid2refs = {}
        for s in splits:
            self.uid2refs.update(read_phn(os.path.join(phn_dir, s + ".tsv")))

        # Remove files that are too long or too short
        logger.info(
            f"Removing files shorter than {min_audio_len} or longer than {max_audio_len} frames"
        )
        orig_tot_len = sum([l for _, l in data_list]) / sample_rate / 3600.0
        orig_num_file = len(data_list)
        data_list = [
            (p, l) for p, l in data_list if l >= min_audio_len and l <= max_audio_len
        ]
        new_tot_len = sum([l for _, l in data_list]) / sample_rate / 3600.0
        logger.info(f"Original audio files: {orig_num_file} ({orig_tot_len:.2f} hrs)")
        logger.info(f"Final audio files:    {len(data_list)} ({new_tot_len:.2f} hrs)")

        # Sort by length (long to short)
        data_list = sorted(data_list, key=lambda x: x[1], reverse=True)

        # Extract speaker
        spk_list = [Path(p).stem.split("-")[0] for p, _ in data_list]
        name_list = [Path(p).stem for p, _ in data_list]
        self.data = [
            (p, l, s, n) for (p, l), s, n in zip(data_list, spk_list, name_list)
        ]
        self.data_lens = [l for _, l, _, _ in self.data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[torch.FloatTensor]:
        # each sample: (audio file path, number of samples, speaker id)
        path, num_frames, spk, uid = self.data[index]

        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)

        wav = wav.astype(np.float32)

        # Crop wave length
        if self.crop_len > 0 and len(wav) > self.crop_len:
            wav = wav[: self.crop_len]

        wav = torch.from_numpy(wav)

        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)

        return wav, uid


def collate_fn(
    batch: List[List[Tuple[torch.FloatTensor, List[int]]]],
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.BoolTensor,]:
    wav_list = []
    wav_len = []

    for wavs in batch:
        for _, w in enumerate(wavs):
            wav_list.append(w)
            wav_len.append(len(w))

    wav_list = pad_sequence(wav_list, batch_first=True)
    wav_len = torch.LongTensor(wav_len)
    padding_mask = len_to_padding(wav_len)
    # padding_mask: has value <=> 0 else 1

    return wav_list, wav_len, padding_mask


def val_collate_fn(
    batch: List[List[Tuple[torch.FloatTensor, List[int]]]],
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
    wav_list = []
    wav_len = []
    uid_list = []

    for wav, uid in batch:
        wav_list.append(wav)
        wav_len.append(len(wav))
        uid_list.append(uid)

    wav_list = pad_sequence(wav_list, batch_first=True)
    wav_len = torch.LongTensor(wav_len)
    padding_mask = len_to_padding(wav_len)
    # padding_mask: has value <=> 0 else 1

    return wav_list, wav_len, padding_mask, uid_list
