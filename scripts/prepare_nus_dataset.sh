# Description: This script downloads the Nus dataset and prepares it for training.

DATASET_DIR_NAME="dataset_nus"
mkdir -p $DATASET_DIR_NAME

# Check if the dataset_nus is already processed
if [ -f "$DATASET_DIR_NAME/DONE" ]; then
    echo "$DATASET_DIR_NAME already processed"
    exit 0
fi

set -e
set -x

# Function to download the dataset_nus
function downloadNus() {
    # Check if download is needed
    if [ -f "Nus.zip" ]; then
        echo "Dataset already downloaded"
    else
        echo "Downloading dataset"
        gdown 1lGHfVN4jWh-oWKpxnQIwZux1qpk5L9C5
    fi
    mv Nus.zip $DATASET_DIR_NAME
    cd $DATASET_DIR_NAME/
    set +e
    unzip Nus.zip && rm Nus.zip && mkdir raw && mv * raw/
    set -e
    cd ..
}

# Function to downsample audios
function downsample() {
    python3 scripts/downsample.py \
        --in-audio-format wav \
        --in-dir $DATASET_DIR_NAME/raw \
        --out-dir $DATASET_DIR_NAME/16k \
        --sample-rate 16000 \
        --num-workers 8
}

# Function to create train and test splits
function create_splits() {
    python3 scripts/preprocess_flist.py \
        --source-dir $DATASET_DIR_NAME/16k  \
        --train-list $DATASET_DIR_NAME/train.csv \
        --val-list $DATASET_DIR_NAME/val.csv \
        --test-list $DATASET_DIR_NAME/test.csv \
        --seed 1
}

# Function to extract features
function extract_features() {
    python3 scripts/preprocess_spk.py \
        --in-dir $DATASET_DIR_NAME/16k \
        --out-dir $DATASET_DIR_NAME/spk_embeddings \
        --num-workers 8

    python3 scripts/preprocess_content.py \
        --in-dir $DATASET_DIR_NAME/16k \
        --out-dir $DATASET_DIR_NAME/ssl_features 

    python3 scripts/preprocess_sr.py \
        --in-dir $DATASET_DIR_NAME/16k \
        --wav-dir $DATASET_DIR_NAME/sr \
        --ssl-dir $DATASET_DIR_NAME/ssl_features \
        --num-workers 1

    python3 scripts/preprocess_pitch.py \
        --in-dir $DATASET_DIR_NAME/16k \
        --out-dir $DATASET_DIR_NAME/pitch_features \
        --num-workers 1
    
}

echo "STEP 1"
downloadNus
echo "STEP 2"
downsample
echo "STEP 3"
create_splits
echo "STEP 4"
extract_features
echo "DONE"
rm -rf $DATASET_DIR_NAME/raw
echo "" > $DATASET_DIR_NAME/DONE

set +x
echo "To easily train the model, rename the $DATASET_DIR_NAME folder to 'dataset'"
echo "NOTE: the audios were not cut in small chunks. You might want to do that before training (see segment_vad.py)."
