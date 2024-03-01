import argparse
import os
import random
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default="./dataset/", help="path to source dir")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--all-list", type=str, default="./dataset/all.csv", help="path to all list")
    parser.add_argument("--train-list", default="", help="path to train list")
    parser.add_argument("--val-list", default="", help="path to val list")
    parser.add_argument("--test-list", default="", help="path to test list")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    train = []
    val = []
    test = []
    idx = 0

    data = []
    for language in os.listdir(args.source_dir):
        for speaker in tqdm(os.listdir(os.path.join(args.source_dir, language))):
            for root, dirs, files in os.walk(os.path.join(args.source_dir, language, speaker)):
                for file in files:
                    if file.endswith(".wav"):
                        data.append((os.path.join(root, file), language, speaker))

    shuffle(data)

    print("Writing", args.all_list)
    with open(args.all_list, "w") as f:
        for wavpath, language, speaker in tqdm(data):
            print(wavpath, language, speaker, sep="|", file=f)

    val += data[:int(len(data) * 0.01)]
    test += data[int(len(data) * 0.01):int(len(data) * 0.02)]
    train += data[int(len(data) * 0.02):]

    shuffle(train)
    shuffle(val)
    shuffle(test)

    if args.train_list != "":
        print("Writing", args.train_list)
        with open(args.train_list, "w") as f:
            for wavpath, language, speaker in tqdm(train):
                print(wavpath, language, speaker, sep="|", file=f)

    if args.val_list != "":
        print("Writing", args.val_list)
        with open(args.val_list, "w") as f:
            for wavpath, language, speaker in tqdm(val):
                print(wavpath, language, speaker, sep="|", file=f)

    if args.test_list != "":
        print("Writing", args.test_list)
        with open(args.test_list, "w") as f:
            for wavpath, language, speaker in tqdm(test):
                print(wavpath, language, speaker, sep="|", file=f)
