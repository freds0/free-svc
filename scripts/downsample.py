import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm
from glob import glob


def process(audio_path):
    if os.path.exists(audio_path):
        audio, _ = librosa.load(audio_path, sr=args.sample_rate)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = 0.98 * audio / peak
        save_path = audio_path.replace(args.in_dir, args.out_dir)
        save_path = save_path.replace('.'+args.in_audio_format, '.'+args.out_audio_format)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wavfile.write(
            save_path,
            args.sample_rate,
            (audio * np.iinfo(np.int16).max).astype(np.int16)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--sample-rate", type=int, default=16000, help="sampling rate")
    parser.add_argument("-if", "--in-audio-format", type=str, default="wav", help="audio format of input audios")
    parser.add_argument("-i", "--in-dir", type=str, default="./data", help="path to source dir")
    parser.add_argument("-o", "--out-dir", type=str, default="./", help="path to target dir")
    parser.add_argument("-of", "--out-audio-format", type=str, default="wav", help="audio format of output audios")
    parser.add_argument("-w", "--num-workers", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    filepaths = glob(f'{args.in_dir}/**/*.{args.in_audio_format}', recursive=True)
    if args.num_workers == 1:
        for filename in tqdm(filepaths):
            process(filename)
    else:
        pool = Pool(processes=args.num_workers)
        for _ in tqdm(pool.imap_unordered(process, filepaths)):
            pass
