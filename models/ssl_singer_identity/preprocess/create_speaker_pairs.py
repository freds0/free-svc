import os
import random
import argparse
from pathlib import Path
from librosa.util import find_files
from tqdm import trange


def collect_speaker_ids(roots, speaker_num):
    all_speaker = []
    all_speaker.extend([f.path for f in os.scandir(roots) if f.is_dir()])

    ids = [[speaker.split("/")[-3], speaker.split("/")[-1]] for speaker in all_speaker]

    vox1 = []
    for id in ids:
        if id[0] == roots.split("/")[-2]:
            vox1.append(id[1])

    dev_speaker = random.sample(vox1, k=speaker_num)
    vox1_train = [ids for ids in vox1 if ids not in dev_speaker]

    train_speaker = []

    train_speaker.extend(vox1_train)

    return train_speaker, dev_speaker


def construct_dev_speaker_id_txt(dev_speakers, dev_txt_name):
    f = open(dev_txt_name, "w")
    for dev in dev_speakers:
        f.write(dev)
        f.write("\n")
    f.close()
    return


def sample_wavs_and_dump_txt(root, dev_ids, numbers, meta_data_name):
    wav_list = []
    count_positive = 0
    print(f"generate {numbers} sample pairs")
    for _ in trange(numbers):
        prob = random.random()
        if prob > 0.5:
            dev_id_pair = random.sample(dev_ids, 2)

            # sample 2 wavs from different speaker
            sample1 = "/".join(
                random.choice(find_files(os.path.join(root, dev_id_pair[0]))).split(
                    "/"
                )[-2:]
            )
            sample2 = "/".join(
                random.choice(find_files(os.path.join(root, dev_id_pair[1]))).split(
                    "/"
                )[-2:]
            )

            label = "0"

            wav_list.append(",".join([label, sample1, sample2]))

        else:
            dev_id_pair = random.sample(dev_ids, 1)

            # sample 2 wavs from same speaker
            sample1 = "/".join(
                random.choice(find_files(os.path.join(root, dev_id_pair[0]))).split(
                    "/"
                )[-2:]
            )
            sample2 = "/".join(
                random.choice(find_files(os.path.join(root, dev_id_pair[0]))).split(
                    "/"
                )[-2:]
            )

            label = "1"
            count_positive += 1

            wav_list.append(",".join([label, sample1, sample2]))
    print("finish, then dump file ..")
    f = open(meta_data_name, "w")
    for data in wav_list:
        f.write(data + "\n")
    f.close()

    return wav_list


def group_files(root_folder, select_only_groups=None, no_partitions=False):
    """
    Recursively scan a directory and group files by subdirectory or artist name.

    Args:
    - root_folder: string, the root directory to scan.
    - select_only_groups: list of strings or None, specifies which subdirectories to select.
    - no_partitions: boolean, specifies whether the root directory contains partitions (train/val/test).

    Returns:
    - A dictionary of groups, where the keys are group names (either subdirectory names or artist names) and
      the values are lists of file paths belonging to that group.
    """

    if no_partitions:
        # If there are no partitions, just scan the root directory.
        root_folders = [root_folder]
    else:
        # If thegroup_filere are partitions, scan each partition separately and merge the results.
        root_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder)]

    groups = {}
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".wav"):
                    # Only consider WAV files.
                    group_name = os.path.basename(root)
                    if select_only_groups is None or group_name in select_only_groups:
                        groups.setdefault(group_name, []).append(
                            os.path.join(root, file)
                        )

    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123)
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="path to the dataset directory",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="path to the output directory where sample_pairs.txt will be saved",
    )
    parser.add_argument(
        "-n",
        "--speaker_num",
        default=40,
        help="number of speakers of the dataset",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--sample_pair",
        default=20000,
        type=int,
        help="number of sample pairs to draw",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    # Example: python create_speaker_pairs.py -r /path/to/dataset -o /where/sample_pairs/will/be/saved -n n_singers -p n_draws
    dataset_dir = Path(args.root)
    dataset_name = dataset_dir.name
    metadata_path = Path(args.output_dir) / "metadata" / dataset_name
    os.makedirs(metadata_path, exist_ok=True)

    a = group_files(dataset_dir, no_partitions=False)
    spkrs = list(a.keys())
    assert len(spkrs) == int(
        args.speaker_num
    ), f"Number of speakers scanned {len(spkrs)} is not equal to provided {args.speaker_num}"

    wav_list = sample_wavs_and_dump_txt(
        dataset_dir, spkrs, args.sample_pair, f"{metadata_path}/speaker_pairs.txt"
    )

    print(f"finish, then dump file {metadata_path}/speaker_pairs___.txt ..")
