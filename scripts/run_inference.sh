# !/bin/bash
EXPERIMENT_PATH="/raid/alefiury/svc/free-svc/logs/config-online-language-emb.yaml/2024-02-23/10-11-19"
INPUT_BASE_DIR="/raid/lucasgris/free-svc"

HPFILE=$EXPERIMENT_PATH"/.hydra/config.yaml"
PTFILE=$EXPERIMENT_PATH"/G_00012_0200000.pth"
METADATA_PATH="/raid/lucasgris/free-svc/data/in_domain_transcriptions_weighted_spks.csv"
IGNORE_METADATA_HEADER=true
SPK_EMB_BASE_DIR="/raid/lucasgris/free-svc/data/spk_embeddings"
PITCH_PREDICTOR="rmvpe"
OUT_DIR=$EXPERIMENT_PATH"/audios"
USE_TIMESTAMP=false
CONCAT_AUDIO=false
PITCH_FACTOR=0.9544

python3 scripts/inference.py \
    --hpfile=$HPFILE \
    --ptfile=$PTFILE \
    --input-base-dir=$INPUT_BASE_DIR \
    --metadata-path=$METADATA_PATH \
    --ignore-metadata-header=$IGNORE_METADATA_HEADER \
    --spk-emb-base-dir=$SPK_EMB_BASE_DIR \
    --pitch-predictor=$PITCH_PREDICTOR \
    --out-dir=$OUT_DIR \
    --pitch-factor=$PITCH_FACTOR