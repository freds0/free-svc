import os
import random

from tqdm import tqdm
import pandas as pd

def load_filepaths_and_spk(file_path, split="|"):
    with open(file_path, encoding='utf-8') as f:
        data = [line.strip().split(split) for line in f]
    return data

def main():
    train_metadatas_list = [
        "dataset_pop/en_train.csv",
        "dataset_opensinger/zh_train.csv",
        "dataset_jvs/jp_train.csv"
    ]

    val_metadatas_list = [
        "dataset_pop/en_val.csv",
        "dataset_opensinger/zh_val.csv",
        "dataset_jvs/jp_val.csv"
    ]

    test_metadatas_list = [
        "dataset_pop/en_test.csv",
        "dataset_opensinger/zh_test.csv",
        "dataset_jvs/jp_test.csv"
    ]

    train_metadatas = []
    print("Loading train metadatas...")
    for metadata in train_metadatas_list:
        train_metadatas += load_filepaths_and_spk(metadata)

    val_metadatas = []
    print("Loading val metadatas...")
    for metadata in val_metadatas_list:
        val_metadatas += load_filepaths_and_spk(metadata)

    test_metadatas = []
    print("Loading test metadatas...")
    for metadata in test_metadatas_list:
        test_metadatas += load_filepaths_and_spk(metadata)

    print(len(train_metadatas), len(val_metadatas), len(test_metadatas))

    # shuffle data
    print("Shuffling data...")
    train_metadatas = random.sample(train_metadatas, len(train_metadatas))
    val_metadatas = random.sample(val_metadatas, len(val_metadatas))
    test_metadatas = random.sample(test_metadatas, len(test_metadatas))

    # save to csv with no column names
    print("Writing Training")
    with open("metadata/train.csv", "w") as f:
        for wavpath, speaker in tqdm(train_metadatas):
            print(wavpath, speaker, sep="|", file=f)

    print("Writing Validation")
    with open("metadata/val.csv", "w") as f:
        for wavpath, speaker in tqdm(val_metadatas):
            print(wavpath, speaker, sep="|", file=f)

    print("Writing Testing")
    with open("metadata/test.csv", "w") as f:
        for wavpath, speaker in tqdm(test_metadatas):
            print(wavpath, speaker, sep="|", file=f)


if __name__ == "__main__":
    main()