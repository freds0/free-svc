import os
import argparse
import torch
import librosa
import json
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile

import sys
sys.path.append(os.path.dirname('..'))

import utils
from mel_processing import mel_processing
from models.wavlm import WavLM, WavLMConfig

import logging
logging.getLogger('numba').setLevel(logging.WARNING)


def process(args, audio_path, vocoder, cmodel, wav_dir, ssl_dir, hps):
    basename = os.path.basename(audio_path)
    wav, _ = librosa.load(audio_path, sr=hps.sampling_rate)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    mel = mel_processing.mel_spectrogram_torch(
        wav,
        hps.n_fft,
        hps.num_mels,
        hps.sampling_rate,
        hps.hop_size,
        hps.win_size,
        hps.fmin,
        hps.fmax
    )
    for i in range(args.min, args.max+1):
        ssl_path = os.path.join(ssl_dir, basename.replace(".wav", f"_{i}.pt"))
        if wav_dir is not None:
            wav_path = os.path.join(wav_dir, basename.replace(".wav", f"_{i}.wav"))
            if (os.path.exists(ssl_path) and os.path.exists(wav_path)):
                print(f"{ssl_path} and {wav_path} already exists. Skipping {i}.")
                continue
        else:
            wav_path = None
            if os.path.exists(ssl_path):
                print(f"{ssl_path} already exists. Skipping {i}.")
                continue
        mel_rs = utils.transform(mel, i)
        wav_rs = vocoder(mel_rs)[0][0].detach().cpu().numpy()
        _wav_rs = librosa.resample(wav_rs, orig_sr=hps.sampling_rate, target_sr=args.sr)
        wav_rs = torch.from_numpy(_wav_rs).cuda().unsqueeze(0)
        c = utils.get_content(cmodel, wav_rs)
        torch.save(c.cpu(), ssl_path)
        if wav_path is not None:
            wavfile.write(
                wav_path,
                args.sr,
                _wav_rs
            )

def main(args, filepaths, spk, pbar=True):
    print("Preprocessing {} ...".format(spk))
    print("Loading WavLM for content...")
    checkpoint = torch.load(args.checkpoint)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")

    print("Loading vocoder...")
    vocoder = utils.get_vocoder(0)
    vocoder.eval()
    print("Loaded vocoder.")

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)
    if args.wav_dir is not None:
        spk_wav_out_dir = os.path.join(args.wav_dir, spk)
        os.makedirs(spk_wav_out_dir, exist_ok=True)
    else:
        spk_wav_out_dir = None
    spk_ssl_out_dir = os.path.join(args.ssl_dir, spk)
    os.makedirs(spk_ssl_out_dir, exist_ok=True)
    for i, filepath in enumerate(tqdm(filepaths) if pbar else filepaths):
        if not pbar:
            print(f"[{i+1}/{len(filepaths)}] Processing {filepath}")
        try:
            process(args, filepath, vocoder, cmodel, spk_wav_out_dir,
                    spk_ssl_out_dir, hps)
            print(f"Processed {filepath}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}", file=sys.stderr)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--min", type=int, default=68, help="min")
    parser.add_argument("--max", type=int, default=92, help="max")
    parser.add_argument("--config", type=str, default="models/hifigan/config.json", help="path to config file")
    parser.add_argument("--in-dir", type=str, default="dataset", help="path to input dir")
    parser.add_argument("--wav-dir", type=str, help="path to output wav dir")
    parser.add_argument("--ssl-dir", type=str, default="dataset/ssl_features", help="path to output ssl dir")
    parser.add_argument("--checkpoint", type=str, default="./models/wavlm/WavLM-Large.pt", help="path to checkpoint")
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()
    print(sub_folder_list)
    to_process = []
    for spk in sub_folder_list:
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir):
            continue

        filepaths = glob(f'{in_dir}/**/*.wav', recursive=True)
        if args.num_workers > 1:
            to_process.append((args, filepaths, spk, False))
        else:
            main(args, filepaths, spk)
    if args.num_workers > 1:
        from multiprocessing import Pool
        with Pool(args.num_workers) as p:
            p.starmap(main, to_process)