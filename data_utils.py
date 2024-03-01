import math
import os
import logging
import random
from warnings import warn
from typing import Callable, List, Union

import numpy as np
import librosa
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler
from scipy.io import wavfile

import utils
import models.commons as commons
from utils import load_wav_to_torch, load_dataset_csv
from mel_processing import mel_processing
from models.f0_predictor import get_f0_predictor


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over Sampler for distributed training. It allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. In such a case, each
    process can pass a torch.utils.data.DistributedSampler instance as a torch.utils.data.DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note:
        Dataset is assumed to be of constant size.
    Args:
        sampler: Sampler used for subsampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By default, rank is retrieved
            from the current distributed group.
        shuffle (bool, optional): If True, sampler will shuffle the indices. Default: True.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. This number should be
            identical across all processes in the distributed group. Default: 0.
    Reference: https://github.com/pytorch/pytorch/issues/23430
    """

    def __init__(
        self,
        sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self):
        indices = list(self.dataset)[: self.total_size]

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(
            indices) == self.total_size, f"{len(indices)} != {self.total_size}"

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(
            indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, "generator"):
            self.dataset.generator = torch.Generator().manual_seed(self.seed + epoch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)


class SubsetSampler(Sampler):
    """
    Samples elements sequentially from a given list of indices.

    Args:
        indices (list): a sequence of indices
    """

    def __init__(self, indices):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class PerfectBatchSampler(Sampler):
    """
    Samples a mini-batch of indices for a balanced class batching

    Args:
        dataset_items(list): dataset items to sample from.
        classes (list): list of classes of dataset_items to sample from.
        batch_size (int): total number of samples to be sampled in a mini-batch.
        num_gpus (int): number of GPU in the data parallel mode.
        shuffle (bool): if True, samples randomly, otherwise samples sequentially.
        drop_last (bool): if True, drops last incomplete batch.
    """

    def __init__(
        self,
        dataset_items,
        classes,
        batch_size,
        num_classes_in_batch,
        num_gpus=1,
        shuffle=True,
        drop_last=False,
        label_key="class_name",
    ):
        super().__init__(dataset_items)
        assert (
            batch_size % (num_classes_in_batch * num_gpus) == 0
        ), "Batch size must be divisible by number of classes times the number of data parallel devices (if enabled)."

        label_indices = {}
        for idx, item in enumerate(dataset_items):
            label = item[label_key]
            if label not in label_indices.keys():
                label_indices[label] = [idx]
            else:
                label_indices[label].append(idx)

        if shuffle:
            self._samplers = [SubsetRandomSampler(
                label_indices[key]) for key in classes]
        else:
            self._samplers = [SubsetSampler(
                label_indices[key]) for key in classes]

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dp_devices = num_gpus
        self._num_classes_in_batch = num_classes_in_batch

    def __iter__(self):

        batch = []
        if self._num_classes_in_batch != len(self._samplers):
            valid_samplers_idx = random.sample(
                range(len(self._samplers)), self._num_classes_in_batch)
        else:
            valid_samplers_idx = None

        iters = [iter(s) for s in self._samplers]
        done = False

        while True:
            b = []
            for i, it in enumerate(iters):
                if valid_samplers_idx is not None and i not in valid_samplers_idx:
                    continue
                idx = next(it, None)
                if idx is None:
                    done = True
                    break
                b.append(idx)
            if done:
                break
            batch += b
            if len(batch) == self._batch_size:
                yield batch
                batch = []
                if valid_samplers_idx is not None:
                    valid_samplers_idx = random.sample(
                        range(len(self._samplers)), self._num_classes_in_batch)

        if not self._drop_last:
            if len(batch) > 0:
                groups = len(batch) // self._num_classes_in_batch
                if groups % self._dp_devices == 0:
                    yield batch
                else:
                    batch = batch[: (groups // self._dp_devices) *
                                  self._dp_devices * self._num_classes_in_batch]
                    if len(batch) > 0:
                        yield batch

    def __len__(self):
        class_batch_size = self._batch_size // self._num_classes_in_batch
        return min(((len(s) + class_batch_size - 1) // class_batch_size) for s in self._samplers)


def identity(x):
    return x


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Taken from https://github.com/PetrochukM/PyTorch-NLP

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """

    def __init__(self, data, sort_key: Callable = identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """Bucket batch sampler

    Adapted from https://github.com/PetrochukM/PyTorch-NLP

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        data (list): List of data samples.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.

    Example:
        >>> sampler = WeightedRandomSampler(weights, len(weights))
        >>> sampler = BucketBatchSampler(sampler, data=data_items, batch_size=32, drop_last=True)
    """

    def __init__(
        self,
        sampler,
        data,
        batch_size,
        drop_last,
        sort_key: Union[Callable, List] = identity,
        bucket_size_multiplier=100,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.data = data
        self.sort_key = sort_key
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(sampler))
        self.bucket_sampler = BatchSampler(sampler, _bucket_size, False)

    def __iter__(self):
        for idxs in self.bucket_sampler:
            bucket_data = [self.data[idx] for idx in idxs]
            sorted_sampler = SortedSampler(bucket_data, self.sort_key)
            for batch_idx in SubsetRandomSampler(list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                sorted_idxs = [idxs[i] for i in batch_idx]
                yield sorted_idxs

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)


class FeatureAudioSpeakerLoader(torch.utils.data.Dataset):

    def __init__(self, file_path, config, shuffle=True, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.config = config
        self.logger.setLevel(config.log_level)
        self.logger.info(f"Initializing FeatureAudioSpeakerLoader - file path: {file_path}")

        self.metadata = load_dataset_csv(file_path)
        self.audio_paths = [x[0] for x in self.metadata]
        self.lang = [x[1] for x in self.metadata]
        self.speakers = [x[2] for x in self.metadata]

        if len(self.audio_paths) < config.train.batch_size:
            self.logger.warning(
                f"Number of audio files ({len(self.audio_paths)}) is less than batch size ({config.train.batch_size})."
            )

        self.filter_length = config.data.filter_length
        self.hop_length = config.data.hop_length
        self.max_wav_value = config.data.max_wav_value
        self.pitch_features_dir = config.data.pitch_features_dir
        self.sampling_rate = config.data.sampling_rate
        self.spec_len = config.train.max_speclen
        self.spectrogram_dir = config.data.spectrogram_dir
        self.spk_embeddings_dir = config.data.spk_embeddings_dir
        self.sr_min_max = config.data.sr_min_max
        self.content_feature_dir = config.data.content_feature_dir
        self.use_spk_emb = config.data.use_spk_emb
        self.use_sr = config.train.use_sr
        self.win_length = config.data.win_length

        # Retro-compatibility with previous config files
        self.use_lang_emb = config.data.get("use_lang_emb", False)

        if self.spectrogram_dir is not None:
            self.logger.info(
                f"Creating spectrogram directory {self.spectrogram_dir}")
            os.makedirs(self.spectrogram_dir, exist_ok=True)

        if self.pitch_features_dir is None:
            self.logger.info("pitch_features_dir is None. Will compute pitch features during training.")

        self.logger.info(
            "Loading Pitch Predictor for pitch features...")
        self.pitch_predictor = get_f0_predictor(
            config.data.pitch_predictor,
            sampling_rate=self.sampling_rate,
            hop_length=self.hop_length,
            device='cpu',
            threshold=0.05
        )
        self.logger.info("Loaded Pitch Predictor.")

        random.seed(config.seed)
        if shuffle:
            random.shuffle(self.metadata)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath, _, _ in self.metadata:
            lengths.append(os.path.getsize(
                audiopath) // (2 * self.hop_length))
        self.lengths = lengths

    def _load_audio_norm(self, audio_path):
        audio, sampling_rate = load_wav_to_torch(audio_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def _load_spectrogram(self, audio_path, audio, lang, speaker):
        spec_path = os.path.join(
            self.spectrogram_dir if self.spectrogram_dir is not None else "",
            lang,
            speaker,
            os.path.basename(audio_path).replace(".wav", ".spec.pt")
        )
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            spec = mel_processing.spectrogram_torch(audio,
                                                    self.filter_length,
                                                    self.sampling_rate,
                                                    self.hop_length,
                                                    self.win_length,
                                                    center=False).squeeze(0)
            if self.spectrogram_dir is not None:
                torch.save(spec, spec_path)
        return spec

    def _load_spk_embedding(self, audio_path, speaker):
        if self.spk_embeddings_dir is not None:
            spk_path = os.path.join(
                self.spk_embeddings_dir if self.spk_embeddings_dir is not None else "",
                speaker,
                os.path.basename(audio_path).replace(".wav", ".npy")
            )
            self.logger.debug("/".join([
                self.spk_embeddings_dir if self.spk_embeddings_dir is not None else "",
                speaker,
                os.path.basename(audio_path).replace(".wav", ".npy")
            ]))
            if not os.path.isfile(spk_path):
                raise Exception(f"Speaker embedding not found at {spk_path}. "
                                "Please run preprocess_spk.py to generate speaker embeddings "
                                "or set spk_embeddings_dir to None (will compute during training).")

            spk = torch.from_numpy(np.load(spk_path))
            self.logger.debug(f"Loaded spk.shape: {spk.shape}")

        else:
            return None # Speaker embedding extraction was removed and moved to the forward pass of the model
        return spk

    def _load_content_feature(self, audio_path, lang, speaker):
        if self.content_feature_dir is not None:
            if not self.use_sr:
                c_path = os.path.join(
                    self.content_feature_dir if self.content_feature_dir is not None else "",
                    lang,
                    speaker,
                    os.path.basename(audio_path).replace(".wav", ".pt")
                )
                self.logger.debug("Loading " + "/".join([
                    self.content_feature_dir if self.content_feature_dir is not None else "",
                    lang,
                    speaker,
                    os.path.basename(audio_path).replace(".wav", ".pt")
                ]))
            else:
                i = random.randint(self.sr_min_max[0], self.sr_min_max[1])
                c_path = os.path.join(
                    self.content_feature_dir if self.content_feature_dir is not None else "",
                    lang,
                    speaker,
                    os.path.basename(audio_path).replace(".wav", f"_{i}.pt")
                )
                self.logger.debug("Loading " + "/".join([
                    self.content_feature_dir if self.content_feature_dir is not None else "",
                    lang,
                    speaker,
                    os.path.basename(audio_path).replace(".wav", f"_{i}.pt")
                ]))
            if not os.path.isfile(c_path):
                raise Exception(f"Content feature not found at {c_path}. "
                                "Please run preprocess_content.py to generate content features "
                                "or set content_feature_dir to None (will compute during training).")

            c = torch.load(c_path).squeeze(0)
            self.logger.debug(f"Loaded c.shape: {c.shape}")

        else:
            return None # Content feature extraction was removed and moved to the forward pass of the model
        return c

    def _load_pitch(self, audio_path, audio, lang, speaker):
        if self.pitch_features_dir is not None:
            pitch_path = os.path.join(
                self.pitch_features_dir,
                lang,
                speaker,
                os.path.basename(audio_path).replace(".wav", "_pitch.pt")
            )
            self.logger.debug("Loading " + "/".join([
                self.pitch_features_dir,
                lang,
                speaker,
                os.path.basename(audio_path).replace(".wav", "_pitch.pt")
            ]))
            if os.path.exists(pitch_path):
                pitch = torch.load(pitch_path).squeeze().numpy()
                # Clip to avoid negative values
                pitch = np.clip(pitch, 0, 800)
                self.logger.debug(f"Loaded pitch.shape: {pitch.shape}")
            elif self.pitch_predictor is not None:
                #_, _, _, pitch, _ = self.pitch_predictor(wavfile.read(audio_path)[1], self.sampling_rate)
                pitch = self.pitch_predictor.compute_f0(wavfile.read(audio_path)[1])
                if type(pitch) is tuple:
                    self.logger.warning(f"Pitch feature computation might have failed for {audio_path}")
                    pitch = pitch[0]
                os.makedirs(os.path.dirname(pitch_path), exist_ok=True)
                torch.save(torch.tensor(pitch), pitch_path)
            # else:
            #     raise Exception(f"Pitch feature not found at {pitch_path}. "
            #                     "Please run preprocess_pitch.py to generate pitch features "
            #                     "or set pitch_features_dir to None (will compute during training).")
        elif self.pitch_predictor is not None:
            #_, _, _, pitch, _ = self.pitch_predictor(wavfile.read(audio_path)[1], self.sampling_rate)
            pitch = self.pitch_predictor.compute_f0(wavfile.read(audio_path)[1])
            if type(pitch) is tuple:
                self.logger.warning(f"Pitch feature computation might have failed for {audio_path}")
                pitch = pitch[0]

        # Interpolates to ensures that pitch and z have the same length
        z_len = int(audio.shape[-1] / self.hop_length)
        pitch = torch.nn.functional.interpolate(
            torch.tensor(pitch).unsqueeze(0).unsqueeze(0),
            size=z_len
        ).squeeze().unsqueeze(0)

        return pitch

    def _load_language_id(self, lang):
        lang_id = self.config.data.lang2id[lang]
        return lang_id

    def get_audio_and_features(self, data):
        audio_path, lang, speaker = data
        audio_norm = self._load_audio_norm(audio_path)
        spec = self._load_spectrogram(audio_path, audio_norm, lang, speaker)
        c = self._load_content_feature(audio_path, lang, speaker)
        pitch = self._load_pitch(audio_path, audio_norm, lang, speaker)
        if self.use_spk_emb:
            spk = self._load_spk_embedding(audio_path, lang, speaker)
        if self.use_lang_emb:
            lang_id = self._load_language_id(lang)

        if self.use_spk_emb and not self.use_lang_emb:
            return [c, spec, audio_norm, pitch, spk]
        elif self.use_lang_emb and self.use_spk_emb:
            return [c, spec, audio_norm, pitch, spk, lang_id]
        elif self.use_lang_emb and not self.use_spk_emb:
            return [c, spec, audio_norm, pitch, lang_id]
        else:
            return [c, spec, audio_norm, pitch]

    def __getitem__(self, index):
        return self.metadata[index]

    def __len__(self):
        return len(self.metadata)


class FeatureAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config, dataset, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.hps = config
        self.use_sr = config.train.use_sr
        self.use_spk_emb = config.data.use_spk_emb
        # Retro-compatibility with previous config files
        self.use_lang_emb = config.data.get("use_lang_emb", False)
        self.dataset = dataset

    def __call__(self, batch_files_and_speakers):
        self.logger.debug(str(batch_files_and_speakers)[:1000])

        batch = []
        for fp, lang, spk in batch_files_and_speakers:
            self.logger.debug(f"fp: {fp}, lang: {lang}, spk: {spk}")
            try:
                b = self.dataset.get_audio_and_features((fp, lang, spk))
                b.append(fp)
                batch.append(b.copy())
            except:
                self.logger.error(fp)

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        pitch_lengths = torch.LongTensor(len(batch))
        lang_lengths = torch.LongTensor(len(batch))
        if self.use_spk_emb:
            spks = torch.FloatTensor(len(batch), batch[0][4].size(0))
        else:
            spks = None

        if self.use_lang_emb:
            lang_ids = torch.LongTensor(len(batch))
        else:
            lang_ids = None

        spec_padded = torch.FloatTensor(
            len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        pitch_padded = torch.FloatTensor(
            len(batch), batch[0][3].size(0), max_spec_len)

        if batch[0][0] is not None:  # If content is not None
            c_padded = torch.FloatTensor(
                len(batch), batch[0][0].size(0), max_spec_len)
            c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            pitch = row[3]

            pitch_padded[i, :, :pitch.size(1)] = pitch
            pitch_lengths[i] = pitch.size(1)

            if pitch.size(1) != spec.size(1):
                self.logger.error(
                    f"Pitch and spec are different for {fp}: pitch.size={pitch.size(1)} spec.size={spec.size(1)}. Check the duration, sample rate and channels of the audios.")

            if c is not None:
                if pitch.size(1) != c.size(1):
                    self.logger.debug("pitch.size(1) != c.size(1): " +
                                     f"pitch.size(1)={pitch.size(1)} c.size(1)={c.size(1)}")
                    c = torch.nn.functional.pad(
                        c,
                        (
                            pitch.size(1)-c.size(1),
                            0,
                        ),
                        mode="reflect",
                    )
                c_padded[i, :, :c.size(1)] = c

            if self.use_spk_emb:
                try:
                    spks[i] = row[4]
                except:
                    self.logger.error(str(spks[i]) + " " + str(row[4]))
                if self.use_lang_emb:
                    lang_ids[i] = row[5]
            else:
                if self.use_lang_emb:
                    lang_ids[i] = row[4]
        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + \
            1 else self.hps.train.max_speclen + 1
        wav_seglen = spec_seglen * self.hps.data.hop_length

        spec_padded, ids_slice = commons.rand_spec_segments(
            spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(
            wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)

        pitch_padded = commons.slice_segments(
            pitch_padded, ids_slice, spec_seglen)[:, :, :-1]

        spec_padded = spec_padded[:, :, :-1]
        wav_padded = wav_padded[:, :, :-self.hps.data.hop_length]

        if c is not None:
            c_padded = commons.slice_segments(
                c_padded, ids_slice, spec_seglen)[:, :, :-1]
        else:
            c_padded = None

        if self.use_spk_emb and not self.use_lang_emb:
            return c_padded, spec_padded, wav_padded, pitch_padded, spks
        elif self.use_lang_emb and self.use_spk_emb:
            return c_padded, spec_padded, wav_padded, pitch_padded, spks, lang_ids
        elif self.use_lang_emb and not self.use_spk_emb:
            return c_padded, spec_padded, wav_padded, pitch_padded, lang_ids
        else:
            return c_padded, spec_padded, wav_padded, pitch_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket %
                   total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(
                    len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * \
                (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j *
                                                           self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
