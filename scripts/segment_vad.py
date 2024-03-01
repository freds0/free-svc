import os
import soundfile as sf
import torch
import torchaudio


def read_audio(path):
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0), sr

def resample_wav(wav, sr, new_sr):
    wav = wav.unsqueeze(0)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)
    wav = transform(wav)
    return wav.squeeze(0)

def map_timestamps_to_new_sr(vad_sr, new_sr, timestamps, just_begging_end=False):
    factor = new_sr / vad_sr
    new_timestamps = []
    if just_begging_end and timestamps:
        # get just the start and end timestamps
        new_dict = {"start": int(timestamps[0]["start"] * factor), "end": int(timestamps[-1]["end"] * factor)}
        new_timestamps.append(new_dict)
    else:
        for ts in timestamps:
            # map to the new SR
            new_dict = {"start": int(ts["start"] * factor), "end": int(ts["end"] * factor)}
            new_timestamps.append(new_dict)

    return new_timestamps

def get_vad_model_and_utils(use_cuda=False):
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False)
    if use_cuda:
        model = model.cuda()

    get_speech_timestamps, save_audio, _, _, collect_chunks = utils
    return model, get_speech_timestamps, save_audio, collect_chunks

class AudioVADSplitter:

    def __init__(self, device, out_dir, min_sec, max_sec, threshold, vad_sample_rate=8000):
        self.use_cuda = True if device=="cuda" else False
        self.model_and_utils = get_vad_model_and_utils(use_cuda=self.use_cuda)
        self.vad_sample_rate = vad_sample_rate
        self.out_dir = out_dir
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.threshold = threshold

    def get_new_speech_timestamps(self, audio_path, trim_just_beginning_and_end=False):
        # get the VAD model and utils functions
        model, get_speech_timestamps, _, collect_chunks = self.model_and_utils

        # read ground truth wav and resample the audio for the VAD
        orig_wav, orig_sample_rate = read_audio(audio_path)

        # if needed, resample the audio for the VAD model
        if orig_sample_rate != self.vad_sample_rate:
            wav = resample_wav(orig_wav, orig_sample_rate, self.vad_sample_rate)
        else:
            wav = orig_wav

        if self.use_cuda:
            wav = wav.cuda()

        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=self.vad_sample_rate, window_size_samples=768, threshold=self.threshold
        )

        # map the current speech_timestamps to the sample rate of the ground truth audio
        new_speech_timestamps = map_timestamps_to_new_sr(
            self.vad_sample_rate, orig_sample_rate, speech_timestamps, trim_just_beginning_and_end
        )

        return orig_wav, orig_sample_rate, new_speech_timestamps

    def __call__(self, audio_path, out_dir=None):
        if not out_dir:
            out_dir = self.out_dir
        if not out_dir:
            raise ValueError("Must specify output dir")

        orig_wav, orig_sample_rate, new_timestamps = self.get_new_speech_timestamps(audio_path)
        orig_wav = orig_wav.cpu().numpy()

        prefix = os.path.splitext(os.path.basename(audio_path))[0]
        os.makedirs(os.path.join(out_dir, prefix), exist_ok=True)
        for tp in new_timestamps:
            s = tp["start"]
            e = tp["end"]

            out_path = os.path.join(out_dir, prefix, f"{prefix}_{s}_{e}.wav")
            if (e-s)//orig_sample_rate < self.min_sec:
                print(f"Ignoring {e//orig_sample_rate}:{s//orig_sample_rate} (< {self.min_sec})")
            elif (e-s)//orig_sample_rate > self.max_sec:
                print(f"Ignoring {e//orig_sample_rate}:{s//orig_sample_rate} (> {self.max_sec})")
            else:
                print(f"Saving audio {out_path}")
                sf.write(file=out_path, data=orig_wav[s:e], samplerate=orig_sample_rate, subtype="PCM_16")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--dir")
    parser.add_argument("--out-dir", '-o', required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min-sec", type=float, default=0.5)
    parser.add_argument("--max-sec", type=float, default=8.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if not args.files and not args.dir:
        raise ValueError("Must specify either files or dir")

    spliter = AudioVADSplitter(
        device=args.device,
        out_dir=args.out_dir,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        threshold=args.threshold
    )

    if args.files:
        for file in args.files:
            print('>', file)
            save_dir = os.path.join(args.out_dir, os.path.splitext(os.path.basename(file))[0])
            print("Saving to", save_dir)
            if args.skip_existing and os.path.isdir(save_dir):
                print("Skipping", save_dir)
                continue
            try:
                spliter(file)
            except Exception as e:
                print("Error:", e, file=sys.stderr)
    elif args.dir:
        for spk_dir in os.listdir(args.dir):
            for root, subfolders, files in os.walk(args.dir):
                for file in list(filter(lambda f: f.endswith('wav'), files)):
                    fp = os.path.join(root, file)
                    print('>', fp)
                    save_dir = os.path.join(args.out_dir, os.path.basename(spk_dir))
                    if args.skip_existing and os.path.exists(save_dir):
                        print("Skipping", save_dir)
                        continue
                    print("Saving to", save_dir)
                    try:
                        spliter(fp, out_dir=save_dir)
                    except Exception as e:
                        print("Error:", e, file=sys.stderr)