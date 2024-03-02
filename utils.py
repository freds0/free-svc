import glob
import json
import logging
import os
import subprocess

import numpy as np
import torch
import torchvision
import yaml
from hydra.core.hydra_config import HydraConfig
from models.commons import sequence_mask
from omegaconf import DictConfig, OmegaConf
from scipy.io.wavfile import read
from torch.nn import functional as F

from models import hifigan
from models.wavlm import WavLM, WavLMConfig

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def get(self, key, default=None):
    return getattr(self, key, default)

def get_cmodel(rank, checkpoint_path="wavlm/WavLM-Large.pt"):
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda(rank)
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    return cmodel


def get_content(cmodel, y):
    with torch.no_grad():
        c = cmodel.extract_features(y.squeeze(1))[0]
    c = c.transpose(1, 2)
    return c


def get_vocoder(rank, config="models/hifigan/config.json", ckpt="models/hifigan/generator_v1"):
    with open(config, "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(ckpt)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.cuda(rank)
    return vocoder


def transform(mel, height):
    tgt = torchvision.transforms.functional.resize(mel, (height, mel.size(-1)))
    if height >= mel.size(-2):
        return tgt[:, :mel.size(-2), :]
    else:
        silence = tgt[:, -1:, :].repeat(1, mel.size(-2)-height, 1)
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


def stretch(mel, width):  # 0.5-2
    return torchvision.transforms.functional.resize(mel, (mel.size(-2), width))

def load_weights(model, checkpoint_path, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'], strict=strict)
    return model

def load_checkpoint(checkpoint_path, model, optimizer=None, strict=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'epoch' in checkpoint_dict:
        epoch = checkpoint_dict['epoch']
    else:
        epoch = 1
        logger.info("Epoch information is not found in the checkpoint. Assume it is 1.")
    if 'learning_rate' in checkpoint_dict:
        learning_rate = checkpoint_dict['learning_rate']
    else:
        learning_rate = None
        logger.info("Learning rate information is not found in the checkpoint.")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if strict:
        assert state_dict.keys() == saved_state_dict.keys(
        ), "Mismatched model config and checkpoint."
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (epoch {})" .format(
        checkpoint_path, epoch))
    return model, optimizer, learning_rate, epoch


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path):
    logger.info("Saving model and optimizer state at epoch {} to {}".format(
        epoch, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f.split('_')[-1]))))
    if len(f_list) == 0:
        return None
    x = f_list[-1]
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_dataset_csv(file_path, split="|"):
    with open(file_path, encoding='utf-8') as f:
        data = [line.strip().split(split) for line in f]
    return data


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
            source_dir
        ))
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. {}(saved) != {}(current)".format(
                saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w").write(cur_hash)


def get_hparams_from_file(config_path):
    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    return hparams
