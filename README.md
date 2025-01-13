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
python scripts/inference.py --hpfile path/to/config.yaml \
                   --ptfile path/to/checkpoint.pth \
                   --input-base-dir path/to/input/directory \
                   --metadata-path path/to/metadata.csv \
                   --spk-emb-base-dir path/to/speaker/embeddings \
                   --out-dir path/to/output_directory \
                   [--use-vad] \
                   [--use-timestamp] \
                   [--concat-audio] \
                   [--pitch-factor PITCH_FACTOR]
```

**Parameters:**
- `--hpfile: Path to the configuration YAML file
- --ptfile: Path to the model checkpoint file
- --input-base-dir: Base directory containing source audio files
- --metadata-path: Path to the CSV metadata file
- --spk-emb-base-dir: Directory containing speaker embeddings
- --out-dir: Output directory for converted audio (default: "gen-samples/")

**Optional Parameters:**
- --pitch-predictor: Pitch predictor model type (default: "rmvpe")
- --use-vad: Enable Voice Activity Detection for better segmentation
- --use-timestamp: Add timestamps to output filenames
- --concat-audio: Concatenate all converted segments into a single file
- --pitch-factor: Adjust pitch modification factor (default: 0.9544)
- --ignore-metadata-header: Skip first row of metadata CSV (default: True)

**Metadata CSV Format:**
```
source_path|source_lang|source_speaker|target_speaker|target_lang
./audio/source1.wav|en|speaker1|speaker2|ja
./audio/source2.wav|zh|speaker3|speaker4|en
```

Required columns:

- source_path: Path to source audio file (relative to input-base-dir)
- source_lang: Source language code (e.g., 'en', 'ja', 'zh')
- source_speaker: Source speaker identifier
- target_speaker: Target speaker identifier
- target_lang: Target language code
- transcript: (Optional) Text transcript of the audio

**Output Directory Structure**
The converted audio files will be organized in the following structure:
```
output_dir/
└── metadata_name/
    └── source_lang/
        └── target_lang/
            └── source_speaker/
                └── target_speaker/
                    └── converted_audio.wav
```

### Additional Notes

- **Voice Activity Detection (VAD)**: When VAD is enabled using the `--use-vad` flag, the system performs intelligent speech segmentation on the input audio. It automatically detects and isolates speech segments for processing while maintaining non-speech portions of the audio. Each detected speech segment is processed independently, and the system then reconstructs the full audio by concatenating all segments in their original order. This approach ensures high-quality conversion while preserving the natural rhythm and timing of the original audio.
  
- **Pitch Adjustment**: The system offers precise pitch control through the `--pitch-factor parameter`, which defaults to 0.9544. This factor acts as a multiplier for the output pitch, where values greater than 1 will increase the pitch of the converted voice, while values less than 1 will lower it. Users can fine-tune this parameter to achieve the desired pitch characteristics in the converted audio.
  
- **Audio Concatenation**: The `--concat-audio` option provides a convenient way to combine multiple conversions into a single audio file. When enabled, the system will automatically merge all converted segments into one continuous audio file, saved as "all.wav" in the output directory. This feature is particularly useful when processing multiple short segments that belong together or when creating a compilation of converted audio.
## Pretrained Models

**The pretrained models will be made publicly available in the near future.**

The pretrained weights for FreeSVC are available on the Hugging Face Model Hub at [alefiury/free-svc](https://huggingface.co/alefiury/free-svc)

Pretrained 

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
