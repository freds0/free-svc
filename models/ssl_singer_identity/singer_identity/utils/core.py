import torch
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm


def sim_cos(enc_x, enc_y, temp=1):
    try:
        return cosim(enc_x, enc_y) / temp
    except NameError:
        cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        return cosim(enc_x, enc_y) / temp


def similarity(enc_x, enc_y, temp=1):
    # return distance_dot(enc_x, enc_y, temp)
    return sim_cos(enc_x, enc_y, temp)


def roll(x):
    return torch.cat((x[-1:], x[:-1]))


def get_audio_length(fn):
    # Returns length of audio file in samples
    import soundfile as sf

    with sf.SoundFile(fn, "r") as f:
        return f.frames


def to_numpy(variable):
    if type(variable) == np.ndarray:
        return variable
    try:
        if torch.cuda.is_available():
            return variable.data.cpu().numpy()
        else:
            return variable.data.numpy()
    except:
        try:
            return variable.numpy()
        except:
            print("Could not 'to_numpy' variable of type " f"{type(variable)}")
            return variable


def freeze_params(module: torch.nn.Module):
    """
    Freeze the parameters of a module.
    """
    for p in module.parameters():
        p.requires_grad = False


# ------------------------------------------------------------


def normalize_signal(signal):
    max_abs = np.maximum(np.abs(np.min(signal)), np.max(signal))
    if max_abs > 0:
        signal /= max_abs
    return signal


def get_fragment_from_file(
    fn, nr_samples, normalize=False, from_=0, draw_random=False, sr=44100, verbose=False
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
            if verbose:
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


def prepare_fn_groups_vocal(
    root_folder,
    groups=None,
    select_only_groups=None,
    filter_fun_level1=None,
    group_name_is_folder=True,
    group_by_artist=False,
    verbose=False,
):
    if filter_fun_level1 is None:

        def filter_fun_level1(x):
            return True

    if groups is None:
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
