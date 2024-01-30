DATASET_DIR_NAME="dataset_jvs"

# Check if the dataset_pop is already processed
if [ -f "$DATASET_DIR_NAME/DONE" ]; then
    echo "$DATASET_DIR_NAME already processed"
    exit 0
fi

set -e
set -x

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
        --source-dir $DATASET_DIR_NAME/jp  \
        --train-list $DATASET_DIR_NAME/jp_train.csv \
        --val-list $DATASET_DIR_NAME/jp_val.csv \
        --test-list $DATASET_DIR_NAME/jp_test.csv \
        --seed 1
}

# echo "STEP 1"
# downloadPopBuTFy
# echo "STEP 2"
# create_spk_dirs
echo "STEP 3"
# downsample
echo "STEP 4"
create_splits
echo "DONE"

echo "" > $DATASET_DIR_NAME/DONE

set +x
echo "To easily train the model, rename the $DATASET_DIR_NAME folder to 'dataset'"