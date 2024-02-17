import os
import subprocess
import shutil
import argparse
import tqdm as tqdm


def process_file(file, root, subfolder_name, sub_dir, output_dir, rename=True):
    if file.endswith(".wav"):
        original_file_path = (
            os.path.join(root, subfolder_name, sub_dir, file)
            if sub_dir is not None
            else os.path.join(root, subfolder_name, file)
        )
        # Get the filename without the extension
        filename = os.path.splitext(file)[0]

        if rename:
            # Construct the new filename, Make sure to remove all "," and " " from the filename and replace with "_"
            sub_dir = f"-{sub_dir}" if sub_dir is not None else ""
            new_filename = f"{subfolder_name}{sub_dir}-{filename}.wav"
            new_filename = new_filename.replace(",", "").replace(" ", "")
        else:
            new_filename = file

        # First, copy the file to the output directory

        shutil.copy(
            original_file_path,
            os.path.join(output_dir, new_filename),
        )


def rename_files(dir_name, dataset_name):
    """Flatten the directory structure of the dataset and rename the files"""
    dataset_dir = os.path.join(dir_name, dataset_name)
    print(dataset_dir)

    # Create the output directory if it does not exist
    output_dir = os.path.join(dir_name, f"{dataset_name}_renamed")
    os.makedirs(output_dir, exist_ok=True)

    # Assuming you have two nested folders in your directory structure
    # List directories only
    dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    for dir in tqdm.tqdm(dirs, desc="Renaming files"):
        # Get the name of the subfolder
        subfolder_name = os.path.basename(dir)
        target_subfolder_name = subfolder_name
        # If m4singer, extract the name of the singer from the subfolder name
        # Ex Alto-1#美错 -> Alto-1, Alto-4#安河桥 -> Alto-4
        if "m4singer" in dataset_name:
            target_subfolder_name = subfolder_name.split("#")[0]
        new_subfolder_dir = os.path.join(output_dir, target_subfolder_name)
        os.makedirs(new_subfolder_dir, exist_ok=True)
        for file_or_dir in os.listdir(os.path.join(dataset_dir, dir)):
            if os.path.isdir(os.path.join(dataset_dir, dir, file_or_dir)):
                for file in os.listdir(os.path.join(dataset_dir, dir, file_or_dir)):
                    process_file(
                        file,
                        dataset_dir,
                        subfolder_name,
                        file_or_dir,
                        new_subfolder_dir,
                    )

            else:
                process_file(
                    file_or_dir, dataset_dir, subfolder_name, None, new_subfolder_dir
                )


def split_dataset(dataset_root_dir, segment_length, sample_rate, exists_partitions):
    full_output_dir = f"{dataset_root_dir}_split_{segment_length}s"
    print(full_output_dir)

    os.makedirs(full_output_dir, exist_ok=True)

    for root, dirs, files in os.walk(dataset_root_dir):
        for file in tqdm.tqdm(files, desc=f"Cropping files for {root}", leave=False):
            if file.endswith(".wav"):
                # Create a directory for each speaker
                speaker_dir = os.path.join(full_output_dir, os.path.basename(root))
                os.makedirs(speaker_dir, exist_ok=True)

                output_file_name = os.path.join(speaker_dir, file)
                # Convert the sample rate to sample_rate
                subprocess.run(
                    [
                        "sox",
                        os.path.join(root, file),
                        "-r",
                        str(sample_rate),
                        output_file_name,
                        "-q",
                    ]
                )

                duration = float(
                    subprocess.check_output(["soxi", "-D", output_file_name])
                )
                segments = int(duration / segment_length)
                for i in range(segments):
                    start = i * segment_length
                    end = (i + 1) * segment_length
                    if duration >= end:
                        segment_name = f"{os.path.splitext(file)[0]}_{start}_{end}_{segment_length}s.wav"
                        subprocess.run(
                            [
                                "sox",
                                output_file_name,
                                os.path.join(speaker_dir, segment_name),
                                "trim",
                                str(start),
                                str(end - start),
                            ]
                        )

                # Remove the original file
                os.remove(output_file_name)


def flatten_dir(full_dir, output_dir):
    """Flatten the directory structure of the dataset. will move files from Dataset/level1/level2/level3/file.wav to Dataset/level1/file.wav"""
    # Assuming you have two nested folders in your directory structure
    # List directories only
    level1_dirs = [
        d for d in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, d))
    ]
    for level1 in level1_dirs:
        # Create the output directory if it does not exist
        os.makedirs(os.path.join(output_dir, level1), exist_ok=True)
        # Move two levels down and get all the files
        for level2 in os.listdir(os.path.join(full_dir, level1)):
            if not os.path.isdir(os.path.join(full_dir, level1, level2)):
                continue
            for level3 in os.listdir(os.path.join(full_dir, level1, level2)):
                if not os.path.isdir(os.path.join(full_dir, level1, level2, level3)):
                    continue
                for file in os.listdir(os.path.join(full_dir, level1, level2, level3)):
                    # Move the file to the level1 directory
                    if file.endswith(".wav"):
                        shutil.copy(
                            os.path.join(full_dir, level1, level2, level3, file),
                            os.path.join(output_dir, level1, file),
                        )


def preprocess_dataset(
    dataset_root_dir,
    dataset_name,
    n_folds,
    segment_length,
    speaker_num,
    sample_rate,
    metadata_path_dir,
    clean=False,
):
    # Create a new dataset folder with the renamed files
    new_dataset_path = os.path.join(dataset_root_dir, f"{dataset_name}_renamed")
    print("renaming files")
    if "vocalset" in dataset_name:
        flatten_dir(os.path.join(dataset_root_dir, dataset_name), new_dataset_path)

    else:
        # Rename files in the dataset to remove "," and " " from the filename and replace with "_"
        rename_files(dataset_root_dir, dataset_name)

    print("renaming done. splitting files now")
    print(new_dataset_path)

    # Split the dataset into segments of n seconds
    split_dataset(new_dataset_path, segment_length, sample_rate, False)
    final_path = f"{new_dataset_path}_split_{segment_length}s"
    print(final_path)

    # Optionally, delete the old dataset renamed folder
    if clean:
        shutil.rmtree(new_dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (also the name of the folder in the dataset root directory)",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        required=True,
        help="Length of segments to split the audio files into",
        default=4,
    )
    parser.add_argument(
        "--sample_rate", type=int, required=True, help="Sample rate of the audio files"
    )
    parser.add_argument(
        "--metadata_path_dir",
        type=str,
        required=False,
        help="Path to the metadata directory",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean the dataset by removing the renamed files",
    )
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset_root_dir,
        args.dataset_name,
        args.segment_length,
        args.sample_rate,
        args.metadata_path_dir,
        args.clean,
    )
