# Description: This script downloads the PopBuTFy dataset and prepares it for training.

echo "This dataset has issues with some audio files."

DATASET_DIR_NAME="dataset_pop"
mkdir -p $DATASET_DIR_NAME

# Check if the dataset_pop is already processed
if [ -f "$DATASET_DIR_NAME/DONE" ]; then
    echo "$DATASET_DIR_NAME already processed"
    exit 0
fi

set -e
set -x

# Function to download the dataset_pop
function downloadPopBuTFy() {
    # Check if download is needed
    if [ -f "PopBuTFy.zip" ]; then
        echo "Dataset already downloaded"
    else
        echo "Downloading dataset"
        gdown 1WQOTrQDVgBeULUWMtBCAhWmiy2fe3hhh
    fi
    mv PopBuTFy.zip $DATASET_DIR_NAME/
    cd $DATASET_DIR_NAME/
    unzip PopBuTFy.zip && rm PopBuTFy.zip 
    cd ..
}

# Function to create spk dirs
function create_spk_dirs() {
    cd $DATASET_DIR_NAME/data/
    set +e
    for i in {10..18}; do
        mkdir Female${i}
        mv "Female${i}#"* Female${i}/
    done

    for i in {1..9}; do
        mkdir Female${i}
        mv "Female${i}#"* Female${i}/
    done

    for i in {1..6}; do
        mkdir Male${i}
        mv "Male${i}#"* Male${i}/
    done
    set -e
    cd ../..
}

# Function to downsample audios
function downsample() {
    python3 scripts/downsample.py \
        --in-audio-format mp3 \
        --in-dir $DATASET_DIR_NAME/data \
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
        --num-workers 4

    python3 scripts/preprocess_pitch.py \
        --in-dir $DATASET_DIR_NAME/16k \
        --out-dir $DATASET_DIR_NAME/pitch_features \
        --num-workers 1
    
}

echo "STEP 1"
downloadPopBuTFy
echo "STEP 2"
create_spk_dirs
echo "STEP 3"
downsample
echo "STEP 4"
create_splits
echo "STEP 5"
extract_features
echo "DONE"
rm -rf dataset_pop/data
echo "" > $DATASET_DIR_NAME/DONE

set +x
echo "To easily train the model, rename the $DATASET_DIR_NAME folder to 'dataset'"