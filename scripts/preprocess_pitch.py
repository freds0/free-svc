import os
import sys
import argparse
import torch
import pyreaper
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
import concurrent.futures


def extract_pitch(file_path, out_dir):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sampling_rate, audio = wavfile.read(file_path)
    print(file_path, sampling_rate)
    save_name = os.path.basename(file_path).replace(".wav", "_pitch.pt")
    save_path = os.path.join(out_dir, save_name)
    if os.path.isfile(save_path):
        print("> Igored because it is already computed: ", save_name)
    else:
        sampling_rate, audio = wavfile.read(file_path)
        print("Audio: ", audio.shape)
        try:
            _, _, _, pitch, _ = pyreaper.reaper(audio, sampling_rate, minf0=50, maxf0=sampling_rate//2)
            print("Pitch: ", pitch.shape)
            torch.save(torch.tensor(pitch), save_path)
        except Exception as e:
            print("> Error in pyreaper: ", file_path, file=sys.stderr)
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="dataset/", help="path to input dir")
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="dataset/pitch_features", help="path to output dir")
    args = parser.parse_args()
    
    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()
    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir):
            continue

        file_paths = glob(f'{in_dir}/**/*.wav', recursive=True)
        spk_out_dir = os.path.join(args.out_dir, spk)
        os.makedirs(spk_out_dir, exist_ok=True)
        
        if args.num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(args.num_workers) as \
                    executor:
                futures = [executor.submit(extract_pitch, file_path, spk_out_dir) for file_path in file_paths]
                for f in tqdm(concurrent.futures.as_completed(futures)):
                    if f.exception() is not None:
                        print(f.exception())
        else:
            for file_path in tqdm(file_paths):
                extract_pitch(file_path, spk_out_dir)
    