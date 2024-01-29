from models.wavlm import WavLM, WavLMConfig
from models.speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
import utils
import argparse
import glob
import logging
import os
import time

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
sys.path.append('..')

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
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=False)
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
        wav_vad, model, sampling_rate=vad_sample_rate, window_size_samples=768)

    # map the current speech_timestamps to the sample rate of the ground truth audio
    new_speech_timestamps = map_timestamps_to_new_sr(
        vad_sample_rate, gt_sample_rate, speech_timestamps
    )
    return new_speech_timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, help="path to yaml config file", required=True)
    parser.add_argument("--ptfile", type=str, help="path to pth file", required=True)
    parser.add_argument("--reference", type=str, help="path to txt file", required=True)
    parser.add_argument("--in-dir", type=str, default="datasets/", help="path to input dir")
    parser.add_argument("--out-dir", type=str, default="gen-samples/", help="path to output dir")
    parser.add_argument("--use-timestamp", default=False, action="store_true")
    parser.add_argument('-pf', "--pitch-factor", default=0.9544, type=float)
    args = parser.parse_args()

    vad_model_and_utils = get_vad_model_and_utils(use_cuda=True)

    os.makedirs(args.out_dir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(
            'speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    srcs = glob.glob(f'{args.in_dir}/**/*.wav', recursive=True)
    srcs.sort()
    print("Processing text...")
    tgts = []
    for f in srcs:
        tgts.append(args.reference)

    print("Synthesizing...")
    all_audios = []
    with torch.no_grad():
        for line in tqdm(zip(srcs, tgts)):
            print(line)
            src, tgt = line
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # get source audios
            speech_frames = return_speech_segments(
                vad_model_and_utils, src, use_cuda=True)
            
            # src
            wav_src_all, _ = librosa.load(src, sr=hps.data.sampling_rate)
            if not speech_frames:
                speech_frames = [{"start": 0, "end": len(wav_src_all)-1}]
            slice_audios = []
            for i in range(len(speech_frames)):
                start = speech_frames[i]["start"]
                end = speech_frames[i]["end"]
                wav_src = wav_src_all[start:end]
                temp_audio = "/tmp/temp_seg_audio"+str(i)+".wav"
                write(temp_audio, hps.data.sampling_rate,
                      (wav_src * 32767).astype(np.int16))
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
                # get pitch
                sampling_rate, audio = wavfile.read(temp_audio)
                _, _, _, pitch, _ = pyreaper.reaper(audio, sampling_rate)
                pitch = np.clip(pitch, 0, 800) * args.pitch_factor
                # interpolat to ensures that pitch and z have the same len
                z_len = round(audio.shape[-1] / hps.data.hop_length)
                pitch = torch.nn.functional.interpolate(torch.tensor(pitch).unsqueeze(0).unsqueeze(
                    0), size=z_len, mode="nearest").squeeze().unsqueeze(0).unsqueeze(0).cuda()

                audio = net_g.infer(c=None, g=g_tgt, mel=mel_tgt, pitch=pitch)
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

            audio = np.concatenate(slice_audios)
            print("Original audio:", len(wav_src_all),
                  "Output audio:", len(audio))
            save_path = src.replace(args.in_dir, args.out_dir)
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(save_path.replace(".wav", "_"+str(timestamp) +
                      ".wav"), hps.data.sampling_rate, audio)
            else:
                write(save_path, hps.data.sampling_rate, audio)
            all_audios.append(audio)

    audio = np.concatenate(all_audios)
    save_path = os.path.join(os.path.dirname(save_path), "all.wav")
    write(save_path, hps.data.sampling_rate, audio)
    print("All audio is saved at:", save_path)
