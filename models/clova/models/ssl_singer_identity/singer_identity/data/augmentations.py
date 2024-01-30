from audiomentations import (
    AddGaussianNoise,
    TimeStretch,
    Shift,
    Gain,
    TanhDistortion,
    SevenBandParametricEQ,
    SevenBandParametricEQ,
    TimeMask,
    ApplyImpulseResponse,
)

from audiomentations.core.transforms_interface import BaseWaveformTransform
import parselmouth
import random

# import librosa
import numpy as np

# from random import random
import warnings
import math

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def aug(signal, augmentations_dict, override=False, sample_rate=44100):
    """Main augmentation function"""
    # Augment the signal

    sig_aug = signal
    # If its not false it is a dict containing manual transforms to apply
    if override is not False:
        for transform in override.values():
            sig_aug = transform(sig_aug)
        return sig_aug
    if augmentations_dict is False:
        return signal
    transforms = aug_factory(augmentations_dict)

    for transform in transforms:
        sig_aug = transform(sig_aug, sample_rate=sample_rate)
    return sig_aug


def aug_factory(augmentation):
    augmentations = []

    if augmentation.get("gaussian_noise", 0):
        augmentations.append(
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.05,
                p=augmentation["gaussian_noise"],
            )
        )

    if augmentation.get("time_stretch", 0):
        augmentations.append(
            TimeStretch(min_rate=0.8, max_rate=1.2, p=augmentation["time_stretch"])
        )

    if augmentation.get("pitch_shift_naive", 0):
        # augmentations.append(PitchShift(min_semitones=-3, max_semitones=3,
        # p=augmentation["time_stretch"]))
        # n_steps = random.choice((-4, 4, 3, -3))
        # ps = lambda x, sample_rate=44100: np.cast["float32"](
        #     librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=n_steps)
        # )
        # augmentations.append(ps)
        pass

    if augmentation.get("formant_shift_parselmouth_prob", 0):
        # if augmentation.get("formant_shift_parselmouth", 0):
        augmentations.append(
            FormantShiftParselmouth(
                augmentation["formant_shift_parselmouth"],
                p=augmentation["formant_shift_parselmouth_prob"],
            )
        )

    if augmentation.get("pitch_shift_parselmouth_prob", 0):
        if augmentation.get("pitch_shift_parselmouth", 0):
            pitch_shift_ratio = augmentation["pitch_shift_parselmouth"]
        else:
            pitch_shift_ratio = 1

        if augmentation.get("pitch_range_parselmouth", 0):
            pitch_range_ratio = augmentation["pitch_range_parselmouth"]
        else:
            pitch_range_ratio = 1

        augmentations.append(
            PitchShiftParselmouth(
                pitch_shift_ratio,
                pitch_range_ratio,
                p=augmentation["pitch_shift_parselmouth_prob"],
            )
        )

    if augmentation.get("shift", 0):
        augmentations.append(
            Shift(
                min_fraction=-0.05,
                max_fraction=0.2,
                p=augmentation["shift"],
                rollover=True,
                fade=True,
            )
        )

    if augmentation.get("gain", 0):
        augmentations.append(
            Gain(min_gain_in_db=-6, max_gain_in_db=0, p=augmentation["gain"])
        )

    if augmentation.get("parametric_eq", 0):
        augmentations.append(
            SevenBandParametricEQ(
                min_gain_db=-2, max_gain_db=1, p=augmentation["parametric_eq"]
            )
        )

    if augmentation.get("tanh_distortion", 0):
        augmentations.append(
            TanhDistortion(
                min_distortion=0.1,
                max_distortion=0.2,
                p=augmentation["tanh_distortion"],
            )
        )

    if augmentation.get("time_mask", 0):
        augmentations.append(TimeMask(max_band_part=1 / 8, p=augmentation["time_mask"]))

    if augmentation.get("reverb", 0):
        ir_path = augmentation["reverb_path"]
        warnings.filterwarnings(
            "ignore", message=".* had to be resampled from 16000 hz to 44100 hz.*"
        )
        augmentations.append(ApplyImpulseResponse(ir_path, p=augmentation["reverb"]))

    return augmentations


class PitchShiftParselmouth(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    def __init__(self, pitch_ratio=1.4, range_ratio=1.3, p=0.5):
        super().__init__(p)

        self.range_ratio = range_ratio
        self.init_range = 1
        self.enable_reciprocal = True
        if type(pitch_ratio) is list:
            self.init_range = float(pitch_ratio[0])
            pitch_ratio = float(pitch_ratio[1])
            # self.enable_reciprocal = True

        self.pitch_ratio = pitch_ratio

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["pitch_shift_ratio"] = random.uniform(
                self.init_range, self.pitch_ratio
            )

            if self.enable_reciprocal:
                use_reciprocal = random.uniform(-1, 1) > 0
                if use_reciprocal:
                    self.parameters["pitch_shift_ratio"] = (
                        1 / self.parameters["pitch_shift_ratio"]
                    )

            self.parameters["pitch_range_ratio"] = random.uniform(1, self.range_ratio)

            use_reciprocal = random.uniform(-1, 1) > 0
            if use_reciprocal:
                self.parameters["pitch_range_ratio"] = (
                    1 / self.parameters["pitch_range_ratio"]
                )

    def apply(self, samples, sample_rate):
        # Add a check to see if samples is numpy array
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
            print("samples is not numpy array, converting to numpy array")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=parselmouth.PraatWarning,
                message="This application uses RandomPool, which is BROKEN in older releases",
            )

            warnings.simplefilter("ignore")
            pitch_shifted_samples = apply_formant_and_pitch_shift(
                wav_to_Sound(samples, sampling_frequency=sample_rate),
                pitch_shift_ratio=self.parameters["pitch_shift_ratio"],
                pitch_range_ratio=self.parameters["pitch_range_ratio"],
                duration_factor=1.0,
            )
        return np.squeeze(np.cast["float32"](pitch_shifted_samples.values))


class FormantShiftParselmouth(BaseWaveformTransform):
    """Formant shift using parselmouth"""

    def __init__(self, formant_shift=1.4, p=0.5):
        super().__init__(p)
        self.init_range = 1
        self.enable_reciprocal = True
        if type(formant_shift) is list:
            self.init_range = float(formant_shift[0])
            formant_shift = float(formant_shift[1])
            self.enable_reciprocal = True

        self.formant_shift = formant_shift

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        if self.parameters["should_apply"]:
            self.parameters["formant_shift_parselmouth"] = random.uniform(
                self.init_range, self.formant_shift
            )

            if self.enable_reciprocal:
                use_reciprocal = random.uniform(-1, 1) > 0
                if use_reciprocal:
                    self.parameters["formant_shift_parselmouth"] = (
                        1 / self.parameters["formant_shift_parselmouth"]
                    )

    def apply(self, samples, sample_rate):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=parselmouth.PraatWarning,
                message="This application uses RandomPool, which is BROKEN in older releases",
            )

            warnings.simplefilter("ignore")
            formant_shifted_samples = apply_formant_and_pitch_shift(
                wav_to_Sound(samples, sampling_frequency=sample_rate),
                formant_shift_ratio=self.parameters["formant_shift_parselmouth"],
                duration_factor=1.0,
            )
        return np.squeeze(np.cast["float32"](formant_shifted_samples.values))


# ------------------------------------------------------------------------------

""" Parselmouth utils for pitch and formant shifting. 
    Part of the code is adapted from https://github.com/dhchoi99/NANSY\
"""


def wav_to_Sound(wav, sampling_frequency: int = 44100) -> parselmouth.Sound:
    r""" load wav file to parselmouth Sound file
    # __init__(self: parselmouth.Sound, other: parselmouth.Sound) -> None \
    # __init__(self: parselmouth.Sound, values: numpy.ndarray[numpy.float6464], 
            sampling_frequency: Positive[float] = 44100.0, start_time: float = 0.0) -> None \
    # __init__(self: parselmouth.Sound, file_path: str) -> None
    returns:
        sound: parselmouth.Sound
    """
    if isinstance(wav, parselmouth.Sound):
        sound = wav
    elif isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav, sampling_frequency=sampling_frequency)
    elif isinstance(wav, list):
        wav_np = np.asarray(wav)
        sound = parselmouth.Sound(
            np.asarray(wav_np), sampling_frequency=sampling_frequency
        )
    else:
        raise NotImplementedError
    return sound


def get_pitch_median(wav, sr: int = None):
    sound = wav_to_Sound(wav, sr)
    pitch = None
    pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    try:
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
        pitch_median = parselmouth.praat.call(
            pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"
        )
    except Exception as e:
        raise e
        pass

    return pitch, pitch_median


def change_gender(
    sound,
    pitch=None,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    new_pitch_median: float = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    try:
        if pitch is None:
            new_sound = parselmouth.praat.call(
                sound,
                "Change gender",
                75,
                600,
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor,
            )
        else:
            new_sound = parselmouth.praat.call(
                (sound, pitch),
                "Change gender",
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor,
            )
    except Exception as e:
        raise e

    return new_sound


def apply_formant_and_pitch_shift(
    sound: parselmouth.Sound,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    """uses praat 'Change Gender' backend to manipulate pitch and formant
    'Change Gender' function: praat -> Sound Object -> Convert -> Change Gender
    see Help of Praat for more details
    # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """

    # pitch = sound.to_pitch()
    pitch = None
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.0:
        try:
            pitch, pitch_median = get_pitch_median(sound, None)
            new_pitch_median = pitch_median * pitch_shift_ratio

            # https://github.com/praat/praat/issues/1926#issuecomment-974909408
            pitch_minimum = parselmouth.praat.call(
                pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"
            )
            newMedian = pitch_median * pitch_shift_ratio
            scaledMinimum = pitch_minimum * pitch_shift_ratio
            resultingMinimum = (
                newMedian + (scaledMinimum - newMedian) * pitch_range_ratio
            )
            if resultingMinimum < 0:
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

            if math.isnan(new_pitch_median):
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

        except Exception as e:
            raise e

    new_sound = change_gender(
        sound,
        pitch,
        formant_shift_ratio,
        new_pitch_median,
        pitch_range_ratio,
        duration_factor,
    )

    return new_sound


def semitones_to_ratio(x):
    return 2 ** (x / 12)
