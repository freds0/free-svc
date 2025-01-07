import argparse
import glob
import logging
import os
import time

import hydra
from hydra.core.hydra_config import HydraConfig
import librosa
import torch
from scipy.io import wavfile
from scipy.io.wavfile import write
from tqdm import tqdm

import numpy as np
import pyreaper
import soundfile as sf
import torch
import torchaudio

import sys
sys.path.append(".")
sys.path.append("..")

import utils
from models.speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from mel_processing import MelProcessing
from models.f0_predictor import get_f0_predictor

from utils import load_dataset_csv


def extract_pitch(pitch_predictor, input_path, output_path):
    pitch = pitch_predictor.compute_f0(wavfile.read(input_path)[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if type(pitch) is tuple:
        print(f"Pitch feature computation might have failed for {input_path}")
        pitch = pitch[0]
    torch.save(torch.tensor(pitch), output_path)

logging.getLogger('numba').setLevel(logging.WARNING)


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


def map_timestamps_to_new_sr(vad_sr, new_sr, timestamps):
    factor = new_sr / vad_sr
    new_timestamps = []
    for ts in timestamps:
        new_dict = {"start": int(ts["start"] * factor),
                    "end": int(ts["end"] * factor)}
        new_timestamps.append(new_dict)

    return new_timestamps


def get_vad_model_and_utils(use_cuda=False):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        onnx=False
    )
    if use_cuda:
        model = model.cuda()

    get_speech_timestamps, save_audio, _, _, collect_chunks = utils
    return model, get_speech_timestamps, save_audio, collect_chunks


def return_speech_segments(
    model_and_utils, audio_path, vad_sample_rate=8000, use_cuda=False
):
    # get the VAD model and utils functions
    model, get_speech_timestamps, _, _ = model_and_utils

    # read ground truth wav and resample the audio for the VAD
    wav, gt_sample_rate = read_audio(audio_path)

    # if needed, resample the audio for the VAD model
    if gt_sample_rate != vad_sample_rate:
        wav_vad = resample_wav(wav, gt_sample_rate, vad_sample_rate)
    else:
        wav_vad = wav

    if use_cuda:
        wav_vad = wav_vad.cuda()

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(
        wav_vad,
        model,
        sampling_rate=vad_sample_rate,
        window_size_samples=768
    )

    # map the current speech_timestamps to the sample rate of the ground truth audio
    new_speech_timestamps = map_timestamps_to_new_sr(
        vad_sample_rate,
        gt_sample_rate,
        speech_timestamps
    )
    return new_speech_timestamps


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
    args = parser.parse_args()

    vad_model_and_utils = get_vad_model_and_utils(use_cuda=True)

    os.makedirs(args.out_dir, exist_ok=True)
    hps = utils.HParams(**utils.get_hparams_from_file(args.hpfile))
    print(hps)

    pitch_predictor = get_f0_predictor(
        args.pitch_predictor,
        sampling_rate=hps.data.sampling_rate,
        hop_length=hps.data.hop_length,
        device="cpu",
        threshold=0.05
    )

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        config=hps
    ).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

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
        if len(rawrow) == 6:
            src, lang_src, speaker_src, transcript, speaker_tgt, lang_tgt = rawrow
        elif len(rawrow) == 5:
            src, lang_src, speaker_src, speaker_tgt, lang_tgt = rawrow
        else:
            raise ValueError(f"Invalid number of columns in metadata: {len(rawrow)}")

        srcs.append(src)
        lang_srcs.append(lang_src)
        speaker_srcs.append(speaker_src)
        speaker_tgts.append(speaker_tgt)
        lang_tgts.append(lang_tgt)

    print("Synthesizing...")
    all_audios = []
    with torch.no_grad():
        for src, lang_src, speaker_src, speaker_tgt, lang_tgt in tqdm(zip(srcs, lang_srcs, speaker_srcs, speaker_tgts, lang_tgts), desc="Synthesizing", total=len(srcs)):
            src = os.path.join(args.input_base_dir, src.replace("./", ""))
            print("Processing:", src)

            # Get source language id if it exists
            if lang2id is None:
                lang_src_id = None
            else:
                lang_src_id = lang2id[lang_src]
                lang_src_id = torch.tensor(lang_src_id).unsqueeze(0).cuda()

            # Get target speaker embedding
            g_tgt_path = os.path.join(args.spk_emb_base_dir, lang_tgt, speaker_tgt+".pt")
            g_tgt = torch.load(g_tgt_path).cuda()

            g_tgt = g_tgt.unsqueeze(-1)

            wav_src_all, _ = librosa.load(src, sr=hps.data.sampling_rate)

            if args.use_vad:
                # Get source audios
                speech_frames = return_speech_segments(
                    vad_model_and_utils,
                    src,
                    use_cuda=True
                )
                if not speech_frames:
                    speech_frames = [{"start": 0, "end": len(wav_src_all)-1}]
                slice_audios = []
                for i in range(len(speech_frames)):
                    try:
                        start = speech_frames[i]["start"]
                        end = speech_frames[i]["end"]
                        wav_src = wav_src_all[start:end]
                        temp_audio = "/tmp/temp_seg_audio"+str(i)+".wav"
                        write(
                            temp_audio,
                            hps.data.sampling_rate,
                            (wav_src * 32767).astype(np.int16)
                        )
                        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
                        # get pitch
                        pitch = pitch_predictor.compute_f0(wav_src_all[start:end])
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
                        if i == 0:
                            if start != 0:
                                slice_audios.append(wav_src_all[:start])
                        else:  # normal samples
                            previous_end = speech_frames[i-1]["end"]
                            if start != previous_end:
                                slice_audios.append(wav_src_all[previous_end:start])

                        slice_audios.append(audio)
                        if i == len(speech_frames)-1:  # last
                            if end != len(wav_src_all)-1:
                                slice_audios.append(wav_src_all[end:])
                    except Exception as e:
                        print(f"Error processing segment {i} of {src}: {e}")
                        raise e
                        slice_audios.append(np.zeros_like(wav_src_all[start:end]))
                        continue

                audio = np.concatenate(slice_audios)
            else:
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

            print(
                "Original audio:", len(wav_src_all),
                "Output audio:", len(audio)
            )
            # save_path = src.replace(args.in_dir, args.out_dir)
            save_path = os.path.join(
                args.out_dir,
                os.path.basename(args.metadata_path.replace(".csv", "")),
                lang_src,
                lang_tgt,
                speaker_src,
                speaker_tgt,
                os.path.basename(src)
            )

            print("Save path:", save_path)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(
                    save_path.replace(".wav", "_"+str(timestamp) +".wav"),
                    hps.data.sampling_rate,
                    audio
                )
            else:
                write(save_path, hps.data.sampling_rate, audio)
            all_audios.append(audio)

    if args.concat_audio:
        audio = np.concatenate(all_audios)
        save_path = os.path.join(os.path.dirname(save_path), "all.wav")
        write(save_path, hps.data.sampling_rate, audio)
        print("All audio is saved at:", save_path)


if __name__ == "__main__":
    main()
