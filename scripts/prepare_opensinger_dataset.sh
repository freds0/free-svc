# Description: This script downloads the PopBuTFy dataset and prepares it for training.

echo "This dataset has issues with some audio files."

DATASET_DIR_NAME="dataset_opensinger"

set -e
set -x

# Function to create spk dirs
function create_spk_dirs() {
    cd $DATASET_DIR_NAME/ManRaw/
    set +e
    for i in {0..9}; do
        mkdir ../Male_${i}
        mv "${i}_"* ../Male_${i}/
    done

    for i in {10..19}; do
        mkdir ../Male_${i}
        mv "${i}_"* ../Male_${i}/
    done

    for i in {20..27}; do
        mkdir ../Male_${i}
        mv "${i}_"* ../Male_${i}/
    done

    cd ../WomanRaw/

    for i in {0..9}; do
        mkdir ../Female_${i}
        mv "${i}_"* ../Female_${i}/
    done

    for i in {10..19}; do
        mkdir ../Female_${i}
        mv "${i}_"* ../Female_${i}/
    done

    for i in {20..29}; do
        mkdir ../Female_${i}
        mv "${i}_"* ../Female_${i}/
    done

    for i in {30..39}; do
        mkdir ../Female_${i}
        mv "${i}_"* ../Female_${i}/
    done

    for i in {40..47}; do
        mkdir ../Female_${i}
        mv "${i}_"* ../Female_${i}/
    done
    set -e
    cd ../..
}

# Function to downsample audios
function downsample() {
    python3 scripts/downsample.py \
        --in-audio-format wav \
        --in-dir $DATASET_DIR_NAME \
        --out-dir $DATASET_DIR_NAME/16k \
        --sample-rate 16000 \
        --num-workers 8
}

# Function to create train and test splits
function create_splits() {
    python3 scripts/preprocess_flist.py \
        --speaker-name-prefix "opensinger_" \
        --source-dir $DATASET_DIR_NAME/zh  \
        --train-list $DATASET_DIR_NAME/zh_train.csv \
        --val-list $DATASET_DIR_NAME/zh_val.csv \
        --test-list $DATASET_DIR_NAME/zh_test.csv \
        --seed 1
}

# echo "STEP 1"
# create_spk_dirs
echo "STEP 2"
# downsample
echo "STEP 3"
create_splits
echo "DONE"
echo "" > $DATASET_DIR_NAME/DONE

set +x
echo "To easily train the model, rename the $DATASET_DIR_NAME folder to 'dataset'"