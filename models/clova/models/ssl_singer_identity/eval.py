import torch
import numpy as np
import pandas as pd
import os
import argparse
import random
from tqdm import tqdm, trange
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from torchaudio import load


from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

aug = None

from singer_identity import load_model, HF_SOURCE


def load_id_extractor(model_file, source):
    """Overwrite load_id_extractor to load your model"""
    raise NotImplementedError


similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)


SIMILARITY_BATCH_SIZE = 512
NUM_WORKERS = 0


def EER(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    s = interp1d(fpr, tpr)
    a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    eer = brentq(a, 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh


class EER_Eval_Dataset(Dataset):
    def __init__(self, file_path, meta_data, reverb=False):
        """Args:
        file_path (string): Path to the data directory.
        meta_data (string): Path to the metadata file containing the speaker pairs.
        reverb (bool): Whether to apply reverb to the audio or not.
        """
        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        # self.vad_c = vad_config
        self.dataset = self.necessary_dict["spk_paths"]
        self.pair_table = self.necessary_dict["pair_table"]
        self.spk_paths = self.necessary_dict["spk_paths"]
        self.reverb = reverb

        self.augs_reverb = {
            "reverb": 0.8,
            "reverb_path": "path_to_RIRs",
        }

    def processing(self):
        pair_table = []
        spk_paths = set()
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split(",")  # [label, path1, path2]
            pair_1 = os.path.join(self.root, list_pair[1]).strip("\n")
            pair_2 = os.path.join(self.root, list_pair[2]).strip("\n")
            spk_paths.add(pair_1)
            spk_paths.add(pair_2)
            one_pair = [list_pair[0], pair_1, pair_2]
            pair_table.append(one_pair)
        return {
            "spk_paths": list(spk_paths),
            "total_spk_num": None,
            "pair_table": pair_table,
        }

    def __len__(self):
        return len(self.necessary_dict["spk_paths"])

    def __getitem__(self, idx):
        x_path = self.dataset[idx]

        x_name = x_path
        nr_samples = 176400  # 4 seconds in 44100 Hz
        try:
            # Load and normalize audio using torchaudio
            wav, _ = load(x_path)
            wav = wav[0]
            # wav = wav.squeeze(0)
            try:
                wav = wav / torch.max(torch.abs(wav))
            except:
                print(
                    f"Warning (get_fragment): could not normalize {x_path}. Returning silence vector"
                )
            if self.reverb:
                wav = aug(wav.numpy(), self.augs_reverb)
                wav = torch.from_numpy(wav)
        except:
            # raise ValueError(f"Could not load {x_path}")
            print(
                f"Warning (get_fragment): could not get fragment from {x_path}. Returning silence vector"
            )
            wav = torch.zeros(nr_samples)

        if wav is None:
            print(
                f"Warning (get_fragment): could not get fragment from {x_path}. Returning silence vector"
            )
            wav = torch.zeros(nr_samples)

        return wav, x_name


class SimilarityEvaluator(torch.nn.Module):
    def __init__(
        self,
        model,
        exp_dir,
        exp_name="experiment",
        use_projection=False,
        compute_eer=True,
        compute_rank=False,
        downsample_upsample=False,
    ):
        super().__init__()
        self.model = model
        self.records = {}
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.use_projection = use_projection
        self._compute_rank_ = compute_rank
        self._compute_eer_ = compute_eer
        self.downsample_upsample = downsample_upsample
        if self.downsample_upsample:
            self.downsample = T.Resample(44100, 16000)
            self.upsample = T.Resample(16000, 44100)

    def forward(self, wav):
        features = self.model.encoder(self.model.feature_extractor(wav))
        if self.use_projection:
            features = self.model.projection(features)
        return features

    @torch.inference_mode()
    def evaluate(self, dataloader):
        device = next(self.model.parameters()).device
        self.model = self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc="Computing embedings"
            ):
                (wavs, others) = batch
                if self.downsample_upsample:
                    wavs = self.downsample(wavs)
                    wavs = self.upsample(wavs)

                wavs = wavs.to(device)
                output = self(wavs)  # Compute embeddings
                utt_name = others  # Vector of filepaths

                for idx in range(len(output)):  # Save embeddings in self.records
                    self.records[utt_name[idx]] = output[idx].cpu()

        eer, rank = None, None
        if self._compute_eer_:
            eer = self.compute_eer(dataloader)
        if self._compute_rank_:
            rank = self.compute_rank(dataloader)

        return {"EER": eer, "rank": rank}

    def compute_rank(self, dataloader):
        trials = dataloader.dataset.pair_table
        speaker_paths = dataloader.dataset.spk_paths
        scores = []
        device = next(self.model.parameters()).device
        MAX_RANK_EVALS = 1000

        # Goes through all pairs in trials but only keeps those that have label 1, saves in new_trials
        # The trials file is not needed for computing rank, only for computing EER,
        # but it is used here because we already have it anyway
        new_trials = []
        for i in range(0, len(trials)):
            label, name1, name2 = trials[i]
            if label == "1":
                new_trials.append(trials[i])

        for i in trange(
            0, MAX_RANK_EVALS, leave=False, desc=f"Rank on {self.exp_name}"
        ):
            label, name1, name2 = new_trials[i]  # q1 and q2 in the paper
            # Finds in speaker_paths all paths that have a different speaker than name1
            # Then compute similarity score between name1 and all other speaker files from different speaker names
            # It is assumed that the first string in the path before "/" is the speaker name
            other_speakers = [
                path for path in speaker_paths if name1.split("/")[-2] not in path
            ]
            # Choose SIMILARITY_BATCH_SIZE random other speakers (S in the paper)
            other_speakers = random.sample(other_speakers, SIMILARITY_BATCH_SIZE)
            other_speakers = [name2] + other_speakers

            # Compute similarity between name1 embeddings saved in self.records and all other speaker files
            # from different speaker names
            other_speakers_embeddings = [
                self.records[path].unsqueeze(0) for path in other_speakers
            ]
            other_speakers_embeddings = torch.cat(other_speakers_embeddings, dim=0).to(
                device
            )
            name1_embeddings = (
                self.records[name1].repeat(len(other_speakers_embeddings), 1).to(device)
            )
            # Compute similarity score between name1 and all other speaker files from different speaker names
            other_speakers_scores = (
                similarity(name1_embeddings, other_speakers_embeddings).cpu().numpy()
            )
            rank = np.argsort(other_speakers_scores)[::-1]
            anchor = np.where(rank == 0)[0][0]
            scores.append(anchor / len(other_speakers_scores))
        mean_rank = np.mean(scores)
        return mean_rank

    def compute_eer(self, dataloader):
        mode = "test"
        trials = dataloader.dataset.pair_table  # Get pairs
        labels = []
        scores = []
        pair_names = []

        # Goes through all pairs and calculates the similarity score
        # Joins pairs in batches of similarity_batch_size to speed up the process on GPU
        for i in tqdm(
            range(0, len(trials), SIMILARITY_BATCH_SIZE),
            leave=False,
            desc=f"EER on {self.exp_name}",
        ):
            batch = trials[i : i + SIMILARITY_BATCH_SIZE]
            batch_name1_embeddings = []
            batch_name2_embeddings = []
            for label, name1, name2 in batch:
                labels.append(label)
                batch_name1_embeddings.append(self.records[name1])
                batch_name2_embeddings.append(self.records[name2])
                pair_names.append(f"{name1.split('/')[-3:]}, {name2.split('/')[-3:]}")

            # Computes the similarity score for all pairs in the batch
            device = next(self.model.parameters()).device
            score = (
                similarity(
                    torch.stack(batch_name1_embeddings).to(device),
                    torch.stack(batch_name2_embeddings).to(device),
                )
                .cpu()
                .numpy()
            )
            score = score.tolist()
            scores.extend(score)

        eer, *others = EER(np.array(labels, dtype=int), np.array(scores))  # eer

        case = "projection" if self.use_projection else "features"

        os.makedirs(Path(self.exp_dir) / self.exp_name, exist_ok=True)
        with open(
            Path(self.exp_dir) / self.exp_name
            # / f"{self.model_epoch}"
            / f"{mode}_predict_{case}.txt",
            "wb",
        ) as file:
            line = [
                f"{name}, {score}\n".encode("utf-8")
                for name, score in zip(pair_names, scores)
            ]
            file.writelines(line)

        with open(
            Path(self.exp_dir) / self.exp_name
            # / f"{self.model_epoch}"
            / f"{mode}_truth_{case}.txt",
            "wb",
        ) as file:
            line = [
                f"{name}, {score}\n".encode("utf-8")
                for name, score in zip(pair_names, labels)
            ]
            file.writelines(line)
        return eer



def append_model_scores(model_name, model_scores, csv_file):
    # If file doesn't exist, create an empty DataFrame with model_name as index
    if not os.path.isfile(csv_file):
        df = pd.DataFrame(index=[model_name])
    else:
        df = pd.read_csv(csv_file, index_col=0)

    for dataset, score in model_scores.items():
        df.loc[
            model_name, dataset
        ] = score  # Update the score for the given model and dataset

    # Save DataFrame back to csv
    df.to_csv(csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123)
    parser.add_argument(
        "-meta",
        "--metadata_dir",
        default="",
        type=str,
        help="path to metadata",
    )
    parser.add_argument(
        "-r",
        "--root",
        default="",
        type=str,
        help="path where datasets are stored",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        nargs="+",
        default=[],
        help="test sets to test on (ex. root_dir/test_set)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="model.ckpt",
        type=str,
        help="mode name or path to model",
    )
    parser.add_argument(
        "-so",
        "--source",
        default=HF_SOURCE,
        type=str,
        help="path to local dir where models are saved or huggingface model space",
    )
    parser.add_argument(
        "-p",
        "--use_projection",
        default=False,
        action="store_true",
        help="wether to compute scores using the projection layer",
    )
    parser.add_argument(
        "-f",
        "--use_features",
        default=True,
        action="store_true",
        help="wehter to compute scores using the encoder feature embeddings",
    )
    parser.add_argument(
        "-cr",
        "--compute_rank",
        default=False,
        action="store_true",
        help="wether to compute rank",
    )
    parser.add_argument(
        "-ce",
        "--compute_eer",
        default=False,
        action="store_true",
        help="only needs to be set when computing rank to also compute EER, If rank is no set, EER will be computed anyway",
    )
    parser.add_argument(
        "-rv",
        "--reverb",
        default=False,
        action="store_true",
        help="wether to use reverb",
    )
    parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "-du",
        "--downsample_upsample",
        default=False,
        action="store_true",
        help="wether to use downsample_upsample",
    )

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_float32_matmul_precision("highest")

    set_seed(args.seed)

    # Check wether whether to use_projection or compute_features is True
    if not args.use_projection and not args.use_features:
        raise ValueError(
            "At least one of use_projection or compute_features must be True"
        )

    root = Path(args.root)

    # Get dataset directories
    test_dataset_names = args.data
    test_dataset_dirs = {
        dataset_name: root / dataset_name for dataset_name in test_dataset_names
    }

    # Paths to speaker pairs file for each dataset
    metadata_dir = Path(args.metadata_dir)
    metadata_paths = {
        dataset_name: metadata_dir / dataset_name for dataset_name in test_dataset_names
    }

    print(f"Running evaluation on {test_dataset_names}")

    # Locate speaker pairs file in metadata dir
    # Make sure that the speaker pairs file is named 'speaker_pairs.txt'
    # preprocess.py will create this file for you
    speaker_pairs = {
        dataset_name: metadata_paths[dataset_name] / "speaker_pairs.txt"
        for dataset_name in test_dataset_names
    }

    test_datasets = {
        dataset_name: EER_Eval_Dataset(
            test_dataset_dirs[dataset_name],
            meta_data=speaker_pairs[dataset_name],
            reverb=args.reverb,
        )
        for dataset_name in test_dataset_names
    }
    test_loaders = {
        dataset_name: DataLoader(
            test_datasets[dataset_name],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        for dataset_name in test_dataset_names
    }

    model_file = args.model
    _compute_eer = True if not args.compute_rank or args.compute_eer else False
    # Get parent folder name if it is a path
    # experiment_name = Path(model_file).parent.name if "/" in model_file else model_file
    experiment_name = model_file
    if args.downsample_upsample:
        experiment_name += "_16khz"
    for test_dataset in test_dataset_names:
        model_scores = {}
        rank_scores = {}

        # Check if model_file is a path
        # If it is a path, set source as the parent folder of model_file and model_file as the basename
        # If it is not a path, set source as the source argument and model_file as the model_file argument
        if "/" in model_file:
            source = Path(model_file).parent
            model_file = Path(model_file).name
        else:
            source = args.source

        try:
            model = load_model(model_file, source)
        except:
            print(f"Could not load model {model_file} from {source}, trying again")
            try:
                model = load_id_extractor(model_file, source)
            except:
                raise ValueError(f"Could not load model {model_file} from {source}")
            
        print(f"Model {model_file} loaded successfully")

        model.eval()
        evaluator = (
            SimilarityEvaluator(
                model,
                exp_dir=metadata_paths[test_dataset],
                exp_name=experiment_name,
                use_projection=args.use_projection,
                compute_rank=args.compute_rank,
                compute_eer=_compute_eer,
                downsample_upsample=args.downsample_upsample,
            )
            .to("cuda") #  Change to "cpu" if you don't have a GPU, but it will be much slower
            .eval()
        )

        test_result = evaluator.evaluate(test_loaders[test_dataset])

        if _compute_eer:
            eer_score = test_result["EER"]
            # round to 4
            eer_score = round(eer_score, 4) * 100
            print(
                f"{experiment_name} score on test dataset {test_dataset}: {eer_score}"
            )
            model_scores[test_dataset] = eer_score
            df = pd.DataFrame.from_dict(model_scores, orient="index")
            # Open existing "eer.csv" file or create a new one and append the new score, do not overwrite
            append_model_scores(
                experiment_name, model_scores, metadata_dir / f"eer_{args.seed}.csv"
            )

        # If rank is computed, save it to a csv file
        if args.compute_rank:
            rank = test_result["rank"] * 100
            rank = round(rank, 4)
            print(f"{experiment_name} rank on test dataset {test_dataset}: {rank}")
            rank_scores[test_dataset] = rank
            rank = pd.DataFrame.from_dict(rank_scores, orient="index")
            # Open existing "rank.csv" file or create a new one and append the new score, do not overwrite
            append_model_scores(
                experiment_name, rank_scores, metadata_dir / f"rank_{args.seed}.csv"
            )
