import argparse
import os
import random
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-name-prefix", type=str, default=None, help="path to output dir")
    parser.add_argument("--source-dir", type=str, default="./dataset/", help="path to source dir")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--train-list", type=str, default="./dataset/train.csv", help="path to train list")
    parser.add_argument("--val-list", type=str, default="./dataset/val.csv", help="path to val list")
    parser.add_argument("--test-list", type=str, default="./dataset/test.csv", help="path to test list")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    speaker_name_prefix = ""
    if args.speaker_name_prefix is not None:
        speaker_name_prefix = args.speaker_name_prefix

    train = []
    val = []
    test = []
    idx = 0

    wavs = []
    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_wavs = []
        for root, dirs, files in os.walk(os.path.join(args.source_dir, speaker)):
            for file in files:
                if file.endswith(".wav"):
                    spk_wavs.append((os.path.join(root, file), speaker_name_prefix+speaker))
        wavs += spk_wavs

    shuffle(wavs)
    val += wavs[:int(len(wavs) * 0.1)]
    test += wavs[int(len(wavs) * 0.1):int(len(wavs) * 0.2)]
    train += wavs[int(len(wavs) * 0.2):]

    shuffle(train)
    shuffle(val)
    shuffle(test)

    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for wavpath, speaker in tqdm(train):
            print(wavpath, speaker, sep="|", file=f)

    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for wavpath, speaker in tqdm(val):
            print(wavpath, speaker, sep="|", file=f)

    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for wavpath, speaker in tqdm(test):
            print(wavpath, speaker, sep="|", file=f)
