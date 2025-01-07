import argparse
import os
from glob import glob

import librosa
import torch
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname('..'))

import utils
from models.wavlm import WavLM, WavLMConfig


def extract_and_save_content_features(audio_path, out_dir, sampling_rate=16000):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    utt_id = os.path.basename(audio_path).rstrip(".wav")
    save_filepath = os.path.join(out_dir, f"{utt_id}.pt")
    if os.path.isfile(save_filepath):
        print("Igored because it is already computed: ", save_filepath)
    else:
        wav, _ = librosa.load(audio_path, sr=sampling_rate)
        wav = torch.from_numpy(wav).unsqueeze(0).cuda()
        c = utils.get_content(cmodel, wav)
        torch.save(c.cpu(), save_filepath)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in-dir", type=str, default="data", help="path to input dir")
    parser.add_argument("--out-dir", type=str, default="data/content_features", help="path to output dir")
    parser.add_argument("--checkpoint", type=str, default="./models/wavlm/WavLM-Large.pt", help="path to checkpoint")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading WavLM for content...")
    checkpoint = torch.load(args.checkpoint)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")

    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()
    for spk in sub_folder_list:
        print("Preprocessing speaker {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir):
            continue

        filepaths = glob(f'{in_dir}/**/*.wav', recursive=True)

        for filepath in tqdm(filepaths):
            spk_out_dir = os.path.join(args.out_dir, spk)
            os.makedirs(spk_out_dir, exist_ok=True)
            extract_and_save_content_features(filepath, spk_out_dir, sampling_rate=args.sr)
