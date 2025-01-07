import json
from pathlib import Path
from typing import List, Tuple

import torchaudio
from tqdm import tqdm


def find_all_librispeech(root: str, sort_by_len: bool = False) -> List[Tuple[str, int]]:
    files = list(Path(root).rglob("*.flac"))
    files = [str(f) for f in files]
    file_lens = [torchaudio.info(f).num_frames for f in tqdm(files)]
    assert len(files) == len(file_lens), (len(files), len(file_lens))
    data = sorted(
        zip(files, file_lens), key=lambda x: x[1 if sort_by_len else 0], reverse=True
    )
    return data


def save_data_info(data: List[Tuple[str, int]], path: str) -> None:
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
