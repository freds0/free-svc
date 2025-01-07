from typing import List, Optional

import torch
import torch.distributed as dist

from .dataset import AudioPretrainDataset


class MaxLengthBatchSampler:
    def __init__(
        self,
        lengths: List[int],
        max_length: int,
        cropped_length: int = 160_000,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 7122,
    ) -> None:
        self.lengths = lengths
        self.max_length = max_length
        self.cropped_length = cropped_length if cropped_length > 0 else 1000000
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        batch_list = []
        batch = []
        cur_length = 0
        for i in range(len(self.lengths)):
            new_batch = batch + [i]
            cur_length += min(self.lengths[i], self.cropped_length)

            if cur_length <= self.max_length:
                batch = new_batch
            elif len(batch) == 0:
                raise ValueError(
                    f"There is a single length {self.lengths[i]} larger than "
                    f"max_length {self.max_length}. Please increase "
                    "the max_length."
                )
            else:
                batch_list.append(batch)
                batch = [i]
                cur_length = min(self.lengths[i], self.cropped_length)

        if len(batch) > 0 and not self.drop_last:
            batch_list.append(batch)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(batch_list), generator=generator).tolist()
        else:
            indices = list(range(len(batch_list)))

        for i in indices:
            yield batch_list[i]

    def __len__(self):
        return len(list(iter(self)))


class MaxLengthDistributedSampler:
    def __init__(
        self,
        dataset: AudioPretrainDataset,
        lengths: List[int],
        max_length: int,
        cropped_length: int = 160_000,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 7122,
        drop_last: bool = False,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        print(f"- Rank: {rank} / Num replicas: {num_replicas}")

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank

        self.lengths = lengths
        self.max_length = max_length
        self.cropped_length = cropped_length if cropped_length > 0 else 1000000

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        batch_list = []
        batch = []
        cur_length = 0
        for i in range(len(self.lengths)):
            new_batch = batch + [i]
            cur_length += min(self.lengths[i], self.cropped_length)

            if cur_length <= self.max_length:
                batch = new_batch
            elif len(batch) == 0:
                raise ValueError(
                    f"There is a single length {self.lengths[i]} larger than "
                    f"max_length {self.max_length}. Please increase "
                    "the max_length."
                )
            else:
                batch_list.append(batch)
                batch = [i]
                cur_length = min(self.lengths[i], self.cropped_length)

        if len(batch) > 0 and not self.drop_last:
            batch_list.append(batch)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(batch_list), generator=generator).tolist()
        else:
            indices = list(range(len(batch_list)))

        max_index = len(indices) - len(indices) % self.num_replicas
        indices = indices[self.rank : max_index : self.num_replicas]
        for i in indices:
            yield batch_list[i]

    def __len__(self):
        return len(list(iter(self)))
