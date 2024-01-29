# FreeSVC: Towards High-Quality Text-Free One-Shot Singing Voice Conversion

## Getting Started

1. Create a docker image from the Dockerfile in this repository.

2. Run the docker image and mount the volume containing this directory.

3. Run the shel script prepare_{name}_dataset.sh to download and preprocess the dataset.

4. Download Wavlm large model from https://github.com/microsoft/unilm/tree/master/wavlm and put it in the directory models/wavlm/;

5. Download HifiGAN model from https://github.com/jik876/hifi-gan and put it in the directory models/hifigan/;

6. Run the following command to train the model:
```
python train.py --config-dir configs --config-name sovits-online_hubert data.dataset_dir={dataset_dir} 
```

7. Convert some audios using the scripts convert.py and convert_dir_vad.py in the scripts directory.

## TODO

- [ ] Test docker image creation again
- [ ] Test conversion scripts