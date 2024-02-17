# Configuration File for Training

You can use a configuration file to train a model using the `train.py` script. Here we provide a description of how to setup the config file. The common options are described in the [common config](common.yaml) file. 


```python
python train.py --config path/to/common.yaml --config path/to/model_config.yaml
```
The model specific options are described below. In the example above, `model_config.yaml` will overwrite the options in `common.yaml` when options are repeated. For more details check the [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) docs.

## 1. Model specific options
In order to use contrastive, VICReg and Uniformity-Alignment, simply change the loss arguments in the config file. Below is the example for the contrastive loss:

```yaml
use_contrastive_loss: true  # use contrastive loss
temp: 0.2  # temperature for contrastive loss
nr_negative: 250  # number of negative samples for contrastive loss
decouple: true  # use decouple contrastive loss or regular NT-Xent loss
use_covariance_reg: false  # use covariance regularization
use_variance_reg: false  # use variance regularization
use_vicreg_loss: false  # use vicreg loss
use_align_loss: false  # use alignment loss
use_uniform_loss: false  # use uniformity loss
```
The individual weights for the losses can be specified as well. BYOL training has its dedicated trainer class and needs to be specified as shown in `byol.yaml`.

We provide the following configs for the models used in the paper:

- `byol.yaml`
- `contrastive.yaml`
- `contrastive_vc.yaml`
- `uniformity-alignment.yaml`
- `vicreg.yaml`


## 2. Data Options
In the config file used to launch training (`common.yaml` is this example), specify the datasets to use as follows:
    
```yaml
data:
class_path: singer_id.data.siamese_encoders.SiameseEncodersDataModule  # default the dataloader class
init_args:
    dataset_dirs: 
    - '/Path/to/dataset1/dataset1_name'
    - '/Path/to/dataset2/dataset2_name'
    batch_size:  # batch size for training
    batch_size_val:  # batch size for validation
    nr_samples: # number of samples to use for training (default: 176000, ie 4 seconds of audio in 44.1kHz)
    normalize: # normalize the audio when loading  
    num_workers: # number of workers for the dataloader
    batch_sampling_mode:  # "sample_clips" or "sample groups". Use "sample_clips" for self-supervised COLA loading
    eval_frac: # fraction of the dataset to use for validation
    group_name_is_folder: 
    group_by_artist: 
    multi_epoch:  # number of epochs to repeat the dataset to simulate a larger dataset
```

## 3. Augmentation Options

The following augmentations are available. We use [Audiomentations](https://github.com/iver56/audiomentations) and [Parselmouth](https://github.com/YannickJadoul/Parselmouth) to perform the augmentations. All fields specify the probability of applying the augmentation, except for `pitch_shift_parselmouth`, `pitch_range_parselmouth`.

```yaml
    augmentations: 
    "enable": true
    "gaussian_noise": 0.5  # min_amplitude=0.001, max_amplitude=0.05
    "pitch_shift_naive": 0  # naive pitch shift (using librosa), not used in the paper
    "time_stretch": 0 # time stretch, not used in the paper
    "gain": 0.5  #  min_gain_in_db=-6, max_gain_in_db=0
    "shift": 0  # not used in the paper
    "parametric_eq": 0  # not used in the paper
    "tanh_distortion": 0  # not used in the paper
    "time_mask": 0.5  # max_band_part=1/8
    "formant_shift_parselmouth": 0  # not used in the paper
    "pitch_shift_parselmouth": [1, 1.3]  # Pitch shift value on parselmouth
    "pitch_range_parselmouth": 1.5  # Pitch range value on parselmouth
    "pitch_shift_parselmouth_prob": 0.5 
```




