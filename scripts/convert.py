from models.wavlm import WavLM, WavLMConfig
from models.speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from mel_processing import mel_processing
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
import torch

import sys
sys.path.append('..')

logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile", type=str, default="configs/freevc.yaml", help="path to yaml config file")
    parser.add_argument(
        "--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--txt-path", type=str,
                        default="convert.txt", help="path to txt file")
    parser.add_argument("--out-dir", type=str,
                        default="output/freevc", help="path to output dir")
    parser.add_argument("--use-timestamp", default=False, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
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

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_processing.mel_spectrogram_torch(
                    wav_tgt,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            # get pitch
            sampling_rate, audio = wavfile.read(src)
            _, _, _, pitch, _ = pyreaper.reaper(audio, sampling_rate)
            pitch = np.clip(pitch, 0, 800) * args.pitch_factor
            # interpolat to ensures that pitch and z have the same len
            z_len = round(audio.shape[-1] / hps.data.hop_length)
            pitch = torch.nn.functional.interpolate(torch.tensor(pitch).unsqueeze(0).unsqueeze(
                0), size=z_len, mode="nearest").squeeze().unsqueeze(0).unsqueeze(0).cuda()

            # TODO: explore other interpolation modes
            c = torch.nn.functional.interpolate(
                c, size=z_len, mode="nearest").cuda()

            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(
                    timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir,
                      f"{title}.wav"), hps.data.sampling_rate, audio)
