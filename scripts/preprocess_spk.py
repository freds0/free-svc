import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname('..'))

from models.speaker_encoder.voice_encoder import SpeakerEncoder
from models.speaker_encoder.audio import preprocess_wav


def build_from_path(in_dir, out_dir, weights_fpath, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wavfile_paths = glob.glob(in_dir + '/**/*.wav', recursive=True)
    wavfile_paths = sorted(wavfile_paths)
    print("Number of wav files: ", len(wavfile_paths))
    if num_workers > 1:
        for wav_path in wavfile_paths:
            futures.append(executor.submit(
                partial(_compute_spkEmbed, out_dir, wav_path, weights_fpath)))
        return [future.result() for future in tqdm(futures)]
    else:
        for wav_path in wavfile_paths:
            _compute_spkEmbed(out_dir, wav_path, weights_fpath)

def _compute_spkEmbed(out_dir, wav_path, weights_fpath):
    utt_id = os.path.basename(wav_path).rstrip(".wav")
    fname_save = os.path.join(out_dir, f"{utt_id}.npy")
    if os.path.isfile(fname_save):
        print("Igored because it is already computed: ", fname_save)
        return os.path.basename(fname_save)
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)

    encoder = SpeakerEncoder(weights_fpath)
    embed = encoder.embed_utterance(wav)
    np.save(fname_save, embed, allow_pickle=False)
    return os.path.basename(fname_save)


def preprocess(in_dir, out_dir, spk, weights_fpath, num_workers):
    out_dir = os.path.join(out_dir, spk)
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, weights_fpath, num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str,
                        default='dataset')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--out-dir', type=str,
                        default='dataset/spk_embeddings')
    parser.add_argument('--spk-encoder-ckpt', type=str,
                        default='models/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()

    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()

    args.num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print("Number of workers: ", args.num_workers)
    ckpt_step = os.path.basename(args.spk_encoder_ckpt).split('.')[0].split('_')[-1]
    spk_embed_out_dir = args.out_dir
    print("[INFO] spk_embed_out_dir: ", spk_embed_out_dir)
    os.makedirs(spk_embed_out_dir, exist_ok=True)

    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir):
            continue
        preprocess(in_dir, spk_embed_out_dir, spk,
                   args.spk_encoder_ckpt, args.num_workers)

    print("DONE!")
    sys.exit(0)
