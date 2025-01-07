<div align="center">

# FreeSVC: Towards Zero-shot Multilingual Singing Voice Conversion

</div>

> **This is the official code implementation of FreeSVC [ICASSP 2025]**

## Introduction

FreeSVC is a multilingual singing voice conversion model that converts singing voices across different languages. It leverages an enhanced VITS model integrated with Speaker-invariant Clustering (SPIN) and the ECAPA2 speaker encoder to effectively separate speaker characteristics from linguistic content. Designed for zero-shot learning, FreeSVC supports cross-lingual voice conversion without the need for extensive language-specific training.

## Key Features

- **Multilingual Support:** Incorporates trainable language embeddings, enabling effective handling of multiple languages without extensive language-specific training.
- **Advanced Speaker Encoding:** Utilizes the State-of-the-Art (SOTA) speaker encoder ECAPA2 to disentangle speaker characteristics from linguistic content, ensuring high-quality voice conversion.
- **Zero-Shot Learning:** Allows cross-lingual singing voice conversion even with unseen speakers, enhancing versatility and applicability.
- **Enhanced VITS Model with SPIN:** Improves content representation for more accurate and natural voice conversion.
- **Optimized Cross-Language Conversion:** Demonstrates the importance of a multilingual content extractor for achieving optimal performance in cross-language voice conversion tasks.

## Model Architecture

FreeSVC builds upon the VITS architecture, integrating several key components:

1. **Content Extractor:** Utilizes SPIN, an enhanced version of ContentVec based on HuBERT, to extract linguistic content while separating speaker timbre.
2. **Speaker Encoder:** Employs ECAPA2 to capture unique speaker characteristics, ensuring accurate disentanglement from linguistic content.
3. **Pitch Extractor:** Uses RMVPE to robustly extract vocal pitches from polyphonic music, preserving the original melody.
4. **Language Embeddings:** Incorporates trainable language embeddings to condition the model for multilingual training and conversion.



<div align="center">
  <img src="resources/freesvc.png" alt="FreeSVC Architecture" height="400">
  <p><em>Figure 1: Comprehensive diagram of the FreeSVC model illustrating the training and inference procedures.</em></p>
</div>

## Dataset

FreeSVC is trained on a diverse set of speech and singing datasets covering multiple languages:

| **Dataset**          | **Hours** | **Speakers** | **Language** | **Type**    |
|----------------------|-----------|--------------|--------------|-------------|
| AISHELL-1            | 170h      | 214 F, 186 M | Chinese      | Speech      |
| AISHELL-3            | 85h       | 176 F, 42 M   | Chinese      | Speech      |
| CML-TTS              | 3.1k      | 231 F, 194 M | 7 Languages  | Speech      |
| HiFiTTS              | 292h      | 6 F, 4 M      | English      | Speech      |
| JVS                  | 30h       | 51 F, 49 M    | Japanese     | Speech      |
| LibriTTS-R           | 585h      | 2,456        | English      | Speech      |
| NUS (NHSS)           | 7h        | 5 F, 5 M      | English      | Both        |
| OpenSinger           | 50h       | 41 F, 25 M    | Chinese      | Singing     |
| Opencpop             | 5h        | 1 F          | Chinese      | Singing     |
| PopBuTFy             | 10h, 40h  | 12, 22        | Chinese, English | Singing |
| POPCS                | 5h        | 1 F          | Chinese      | Singing     |
| VCTK                 | 44h       | 109          | English      | Speech      |
| VocalSet             | 10h       | 11 F, 9 M     | Various      | Singing     |


## Getting Started

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/freds0/free-svc.git
    cd free-svc
    ```

2. **Create a Docker Image:**
    - Build the Docker image using the provided `Dockerfile`:
      ```bash
      docker build -t freesvc .
      ```

3. **Run the Docker Container:**
    - Start the Docker container and mount the current directory:
      ```bash
      docker run -it --rm -v "$(pwd)":/workspace freesvc
      ```

4. **Prepare the Dataset:**
    - Execute the dataset preparation script:
      ```bash
      bash prepare_{name}_dataset.sh
      ```
      Replace `{name}` with the appropriate dataset identifier.

5. **Download Required Models:**
    - **WavLM Large Model:**
      - Download from [WavLM GitHub Repository](https://github.com/microsoft/unilm/tree/master/wavlm).
      - Place the downloaded model in `models/wavlm/`.

    - **HifiGAN Model:**
      - Download from [HifiGAN GitHub Repository](https://github.com/jik876/hifi-gan).
      - Place the downloaded model in `models/hifigan/`.

6. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

7. **Train the Model:**
    - Run the training script with the appropriate configuration:
      ```bash
      python train.py --config-dir configs --config-name sovits-online_hubert data.dataset_dir={dataset_dir}
      ```
      Replace `{dataset_dir}` with the path to your dataset directory.

### Audio Conversion

#### Single Audio File Conversion

```bash
python scripts/convert.py --hpfile path/to/config.yaml \
                         --ptfile path/to/checkpoint.pth \
                         --txt-path path/to/convert.txt \
                         --out-dir path/to/output_dir \
                         [--use-timestamp]
```

**Parameters:**
- `--hpfile`: Configuration file path (Default: configs/freevc.yaml)
- `--ptfile`: Model checkpoint file path (Default: checkpoints/freevc.pth)
- `--txt-path`: Conversion instructions file path
- `--out-dir`: Output directory (Default: output/freevc)
- `--use-timestamp`: (Optional) Add timestamp to output filenames

**Example convert.txt format:**
```
song1|path/to/source1.wav|path/to/target1.wav
song2|path/to/source2.wav|path/to/target2.wav
```

#### Batch Conversion with VAD

```bash
python scripts/convert_dir_vad.py --hpfile path/to/config.yaml \
                                 --ptfile path/to/checkpoint.pth \
                                 --reference path/to/reference.wav \
                                 --in-dir path/to/input_directory \
                                 --out-dir path/to/output_directory \
                                 [--use-timestamp] \
                                 [--concat-audio] \
                                 [--pitch-factor PITCH_FACTOR]
```

**Parameters:**
- `--hpfile`: Configuration file path (Required)
- `--ptfile`: Model checkpoint file path (Required)
- `--reference`: Reference speaker's WAV file path (Required)
- `--pitch-predictor`: Pitch predictor model (Default: rmvpe)
- `--in-dir`: Input WAV files directory (Default: dataset/test/audio)
- `--out-dir`: Output directory (Default: gen-samples/)
- `--use-timestamp`: (Optional) Add timestamp to output filenames
- `--concat-audio`: (Optional) Concatenate all converted segments
- `--pitch-factor`: (Optional) Pitch adjustment factor (Default: 0.9544)

### Additional Notes

- **Voice Activity Detection (VAD)**: The batch conversion script uses VAD to detect and segment speech within input audio files
- **Pitch Adjustment**: Use the pitch factor parameter to control pitch modification during conversion
- **Audio Concatenation**: Enable `--concat-audio` to combine all converted segments into a single file

## Pretrained Models

**The pretrained models will be made publicly available in the near future.**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [**so-vits-svc**](https://github.com/svc-develop-team/so-vits-svc)
- [**VITS Model**](https://github.com/jaywalnut310/vits)
- [**HifiGAN**](https://github.com/jik876/hifi-gan)
- [**ECAPA2 Speaker Encoder**](https://huggingface.co/Jenthe/ECAPA2)
- [**WavLM**](https://github.com/microsoft/unilm/tree/master/wavlm)
- [**ContentVec**](https://github.com/auspicious3000/contentvec)
- [**SPIN**](https://github.com/vectominist/spin)
- [**RMVPE**](https://github.com/Dream-High/RMVPE)
