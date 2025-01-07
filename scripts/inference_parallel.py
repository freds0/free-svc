import sys
sys.path.append(".")
sys.path.append("..")

import os
import time
import glob
import logging
import argparse
import cloudpickle
from joblib import Parallel, delayed, wrap_non_picklable_objects

import torch
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from tqdm_joblib import tqdm_joblib
from scipy.io.wavfile import write

import utils
from utils import load_dataset_csv
from models import SynthesizerTrn
from models.f0_predictor import get_f0_predictor
from models.speaker_encoder.voice_encoder import SpeakerEncoder
from mel_processing import MelProcessing


@wrap_non_picklable_objects
def convert_voice(
    hps,
    args,
    net_g,
    pitch_predictor,
    lang2id,
    src,
    lang_src,
    speaker_src,
    speaker_tgt,
    lang_tgt
):
    net_g = cloudpickle.loads(net_g)
    src = os.path.join(args.input_base_dir, src.replace("./", ""))

    # Get source language id if it exists
    if lang2id is None:
        lang_src_id = None
    else:
        lang_src_id = lang2id[lang_src]
        lang_src_id = torch.tensor(lang_src_id).unsqueeze(0).cuda()
    wav_src_all, _ = librosa.load(src, sr=hps.data.sampling_rate)

    wav_src = torch.from_numpy(wav_src_all).unsqueeze(0).cuda()
    # get pitch
    pitch = pitch_predictor.compute_f0(wav_src_all)
    pitch = np.clip(pitch, 0, 800) * args.pitch_factor
    # interpolat to ensures that pitch and z have the same len
    z_len = round(wav_src.shape[-1] / hps.data.hop_length)
    pitch = torch.nn.functional.interpolate(
        torch.tensor(pitch).unsqueeze(0).unsqueeze(0),
        size=z_len,
        mode="nearest"
    ).squeeze().unsqueeze(0).unsqueeze(0).cuda()

    audio = net_g.voice_conversion(
        c_src=None,
        y_src=wav_src,
        y_tgt=None,
        g_tgt=g_tgt,
        mel_tgt=None,
        c_lengths=None,
        pitch_tgt=pitch,
        lang_id_src=lang_src_id
    )
    audio = audio[0][0].data.cpu().float().numpy()

    save_path = os.path.join(
        args.out_dir,
        os.path.basename(args.metadata_path.replace(".csv", "")),
        lang_src,
        lang_tgt,
        speaker_src,
        speaker_tgt,
        os.path.basename(src)
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write(save_path, hps.data.sampling_rate, audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, help="path to yaml config file", required=True)
    parser.add_argument("--ptfile", type=str, help="path to pth file", required=True)
    parser.add_argument("--input-base-dir", type=str, help="path to input dir", required=True)
    parser.add_argument("--metadata-path", type=str, help="path to metadata file", required=True)
    parser.add_argument("--ignore-metadata-header", type=bool, default=True)
    parser.add_argument("--spk-emb-base-dir", type=str, help="path to reference speaker wav file", required=True)
    parser.add_argument("--pitch-predictor", type=str, default="rmvpe")
    parser.add_argument("--out-dir", type=str, default="gen-samples/", help="path to output dir")
    parser.add_argument("--use-vad", default=False, action="store_true")
    parser.add_argument("--use-timestamp", default=False, action="store_true")
    parser.add_argument("--concat-audio", default=False, action="store_true")
    parser.add_argument('-pf', "--pitch-factor", default=0.9544, type=float)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hps = utils.HParams(**utils.get_hparams_from_file(args.hpfile))
    print(hps)

    net_g = net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        config=hps
    ).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    serialized_object_net_g = cloudpickle.dumps(net_g)

    pitch_predictor = get_f0_predictor(
        args.pitch_predictor,
        sampling_rate=hps.data.sampling_rate,
        hop_length=hps.data.hop_length,
        device="cpu",
        threshold=0.05
    )

    if hasattr(hps.data, "lang2id"):
        lang2id = hps.data.lang2id
    else:
        lang2id = None

    # Load metadata
    srcs, lang_srcs, speaker_srcs, speaker_tgts, lang_tgts = [], [], [], [], []
    rawrows = load_dataset_csv(args.metadata_path)
    # Ignore header if needed
    if args.ignore_metadata_header:
        rawrows = rawrows[1:]
    for rawrow in tqdm(rawrows, desc="Processing metadata"):
        src, lang_src, speaker_src, transcript, speaker_tgt, lang_tgt = rawrow
        srcs.append(src)
        lang_srcs.append(lang_src)
        speaker_srcs.append(speaker_src)
        speaker_tgts.append(speaker_tgt)
        lang_tgts.append(lang_tgt)

    # limit number of samples
    limit_samples = 10
    start = 1000
    srcs = srcs[start:start+limit_samples]
    lang_srcs = lang_srcs[start:start+limit_samples]
    speaker_srcs = speaker_srcs[start:start+limit_samples]
    speaker_tgts = speaker_tgts[start:start+limit_samples]
    lang_tgts = lang_tgts[start:start+limit_samples]

    tasks = [(hps, args, serialized_object_net_g, pitch_predictor, lang2id, src, lang_src, speaker_src, speaker_tgt, lang_tgt) for src, lang_src, speaker_src, speaker_tgt, lang_tgt in zip(srcs, lang_srcs, speaker_srcs, speaker_tgts, lang_tgts)]

    with tqdm_joblib(desc="Preprocessing", total=len(srcs)):
        Parallel(n_jobs=args.num_workers)(delayed(convert_voice)(*task) for task in tasks)


if __name__ == "__main__":
    main()
