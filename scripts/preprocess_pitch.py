import os
import sys
import argparse
import torch
import random
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.f0_predictor import get_f0_predictor

def extract_pitch(pitch_predictor, input_path, output_path, skip_existing=False):
    if skip_existing and os.path.exists(output_path):
        return
    pitch = pitch_predictor.compute_f0(wavfile.read(input_path)[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if type(pitch) is tuple:
        print(f"Pitch feature computation might have failed for {input_path}")
        pitch = pitch[0]
    torch.save(torch.tensor(pitch), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="data/train", help="path to input dir")
    parser.add_argument("--pitch-predictor", type=str, default="rmvpe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", help="skip existing pitch files")
    parser.add_argument("--out-dir", type=str, default="data/pitch_features/train", help="path to output dir")
    args = parser.parse_args()

    if args.device == "cuda" and args.num_workers > 1:
        print("Warning: Multiprocessing with CUDA is not supported. Setting num_workers to 1.")
        args.num_workers = 1

    pitch_predictor = get_f0_predictor(
        args.pitch_predictor,
        sampling_rate=args.sampling_rate,
        hop_length=args.hop_length,
        device=args.device,
        threshold=0.05
    )

    file_paths = glob(f'{args.in_dir}/**/*.wav', recursive=True)
    random.shuffle(file_paths)

    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(args.num_workers) as \
                executor:
            futures = [executor.submit(pitch_predictor, file_path, file_path.replace(args.in_dir, args.out_dir).replace(".wav", "_pitch.pt"), skip_existing=args.skip_existing) for file_path in file_paths]
            for f in tqdm(concurrent.futures.as_completed(futures)):
                if f.exception() is not None:
                    print(f.exception())
    else:
        for file_path in tqdm(file_paths):
            output_path = file_path.replace(args.in_dir, args.out_dir).replace(".wav", "_pitch.pt")
            extract_pitch(pitch_predictor, file_path, output_path, args.skip_existing)