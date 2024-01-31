import soundfile as sf
from tqdm import tqdm
import numpy as np
import os
from utils.core import get_audio_length
import torchaudio
import torch
import random


def normalize_signal(signal):
    max_abs = np.maximum(np.abs(np.min(signal)), np.max(signal))
    if max_abs > 0:
        signal /= max_abs
    return signal


def crop_audio_file(
    file_path, sample_rate, crop_length, hop_size=1, random_crop=False, normalize=False
):
    """
    Reads an audio file and crops it to a specific length in samples, with an optional randomly selected starting position.
    If the audio file has a different sample rate than the desired output, the function will resample the audio.

    Args:
        file_path (str): The path to the audio file.
        crop_length (int): The length in samples to crop the audio to.
        hop_size (int): The hop size in samples. The frame_offset is only selected as an integer multiple of the hop size.
        random_crop (bool): Whether to randomly select the starting position or use the default value of 0.
        sample_rate (int): The desired output sample rate.

    Returns:
        Tensor: A tensor containing the cropped and resampled audio waveform.

    """

    # Get the audio info to obtain the number of samples and sample rate
    info = torchaudio.info(file_path)
    num_samples = info.num_frames
    audio_sample_rate = info.sample_rate
    resample_ratio = audio_sample_rate / sample_rate
    if audio_sample_rate != sample_rate:
        crop_length = int(crop_length * resample_ratio)

    # Compute valid_frames based on the hop size and crop_length
    # Adjust frame_offset if audio_sample_rate is different from the desired sample_rate
    valid_frames = torch.arange(
        0, num_samples - crop_length + 1, int(hop_size * resample_ratio)
    )

    # Determine the valid frame offsets based on the valid_frames list
    valid_offsets = valid_frames.tolist()

    # Determine the frame offset based on whether random is set or not
    if random_crop:
        # Randomly select a frame offset from the valid offsets
        frame_offset = random.choice(valid_offsets)
    else:
        frame_offset = int(
            valid_offsets[-1] / 2
        )  # sometimes the beginnig has a lot of silence

    # if audio_sample_rate != sample_rate:
    #     frame_offset_audio_rate = int(frame_offset * resample_ratio)

    # Load only the desired number of samples starting from the selected offset
    waveform, _ = torchaudio.load(
        file_path, frame_offset=int(frame_offset), num_frames=crop_length
    )

    waveform = torch.mean(waveform, dim=0)
    # Verify that the resulting waveform has the correct number of samples
    if waveform.shape[-1] < crop_length:
        # Pad the waveform with zeros to ensure it has crop_length samples
        pad_amount = crop_length - waveform.shape[-1]
        waveform = torch.nn.functional.pad(
            waveform, (0, pad_amount), mode="constant", value=0
        )

    # Resample the waveform if necessary
    if audio_sample_rate != sample_rate:
        resample_transform = torchaudio.transforms.Resample(
            audio_sample_rate, sample_rate
        )
        waveform = resample_transform(waveform)

    # Normalize the waveform if necessary
    if normalize:
        waveform = normalize_signal(waveform)

    return waveform


def get_fragment_from_file(
    fn, nr_samples, normalize=False, from_=0, draw_random=False, sr=44100
):
    sample = None

    with sf.SoundFile(fn, "r") as f:
        if nr_samples < 0:
            nr_samples = f.frames
        if draw_random:
            draw_interval_size = np.maximum(int(f.frames - nr_samples), 1)
            from_ = np.random.randint(0, draw_interval_size)
        # assert f.samplerate == 44100, f"sample rate is {f.samplerate}, should be 44100 though."
        if f.samplerate != sr:
            print(f"Warning: sample rate is {f.samplerate}, configured as {sr}: {fn}")
        try:
            nr_samples_ = np.minimum(nr_samples, f.frames)
            f.seek(from_)
            sample = f.read(nr_samples_)
            if len(sample.shape) > 1 and sample.shape[1] == 2:
                # to mono
                sample = sample.mean(1)

            # sample too short -> pad
            if len(sample) < nr_samples:
                sample_ = np.zeros((nr_samples,))
                sample_[: len(sample)] = sample
                sample = sample_

            if normalize:
                sample = normalize_signal(sample)
        except Exception as e:
            print(
                f"Warning: could not get fragment from {fn}. Returning silence vector"
            )
            sample = np.zeros((nr_samples,))
            pass
    return sample


def group_files(root_folder, select_only_groups=None, no_partitions=False):
    """
    Recursively scan a directory and group files by subdirectory or artist name.

    Args:
    - root_folder: string, the root directory to scan.
    - select_only_groups: list of strings or None, specifies which subdirectories to select.
    - no_partitions: boolean, specifies whether the root directory contains partitions (train/val/test).

    Returns:
    - A dictionary of groups, where the keys are group names (either subdirectory names or artist names) and
      the values are lists of file paths belonging to that group.
    """

    if no_partitions:
        # If there are no partitions, just scan the root directory.
        root_folders = [root_folder]
    else:
        # If there are partitions, scan each partition separately and merge the results.
        root_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder)]

    groups = {}
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".wav"):
                    # Only consider WAV files.
                    group_name = os.path.basename(root)
                    if select_only_groups is None or group_name in select_only_groups:
                        groups.setdefault(group_name, []).append(
                            os.path.join(root, file)
                        )

    return groups


# this has the same functionality as group_files, but less pretty
def prepare_fn_groups_vocal(
    root_folder,
    groups=None,
    select_only_groups=None,
    filter_fun_level1=None,
    group_name_is_folder=True,
    group_by_artist=False,
    verbose=False,
):
    if filter_fun_level1 == None:

        def filter_fun_level1(x):
            return True

    if groups == None:
        groups = {}

    if not group_by_artist:
        fn_counter = 0
        if verbose:
            print("Not grouping data by subdir")
    else:
        if verbose:
            print("Grouping data by subdir")

    if select_only_groups is not None:
        if verbose:
            print(
                f"Warning: select_only_groups is not None, selecting data only in subdirs {select_only_groups}"
            )
    result = []
    # groups = {}

    for root0, dirs0, _ in tqdm(
        os.walk(root_folder), desc=f"scanning sub-directories of {root_folder}"
    ):
        for dir0 in dirs0:
            if group_name_is_folder:
                group_name = dir0  # Folder
            else:
                group_name = "unknown"
                if select_only_groups is not None:
                    if verbose:
                        print(
                            "Warning: group_name_is_folder is False and select_only_groups is not None"
                        )

            if select_only_groups is None or (
                select_only_groups is not None and group_name in select_only_groups
            ):
                fns_level1 = []
                for root1, dirs1, files1 in os.walk(os.path.join(root0, dir0)):
                    for file1 in files1:
                        fn = os.path.join(root1, file1)
                        if filter_fun_level1(fn):
                            if group_by_artist:
                                if group_name in groups:
                                    groups[group_name].append(fn)
                                else:
                                    groups[group_name] = [fn]
                            else:
                                groups[fn_counter] = [fn]
                                fn_counter += 1
        break  # only looks at first level to find groups
    return groups


def filter1_voice_wav(fn):
    if (
        (fn.endswith("wav") or fn.endswith("WAV") or fn.endswith(".flac"))
        and ".json" not in fn
        and "_mic2" not in fn
    ):
        try:
            if get_audio_length(fn) < (44100 / 10):
                # print(f"too short: {fn}")
                return False
        except RuntimeError:
            print(f"exception: {fn}")
            return False
        # print(f"accept {fn}")
        return True
    return False


def extract_clips_from_fn_list(
    fn_list,
    nr_samples=176400,
    clips_per_fn=1,
    normalize=True,
    draw_random=True,
    flatten=False,
    disable_tqdm=True,
):
    """
    Extract audio clips from list of filepaths
    Args:
        fn_list: List of filepaths
        nr_samples: number of samples to be extracted from clip
        clips_per_fn: number of clips to be drawn from each file
        normalize: whether to normalize or not clip
        draw_random: whether to draw a clip starting at a random location of the file or not
        flatten" wheter to flatten list of lists on output or not
    Returns:
        clips:
    """
    clips = []
    for i in range(clips_per_fn):
        clips_list = [
            np.cast["float32"](
                get_fragment_from_file(
                    fn, nr_samples, normalize=normalize, draw_random=draw_random
                )
            )
            for fn in tqdm(fn_list, desc="extracting clips", disable=disable_tqdm)
        ]
        clips.append(clips_list)
    if flatten:
        clips = [val for sublist in clips for val in sublist]
    return clips
