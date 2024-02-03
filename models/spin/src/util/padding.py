import torch


@torch.no_grad()
def len_to_padding(x_len: torch.LongTensor, max_len: int = 0) -> torch.BoolTensor:
    if max_len == 0:
        max_len = max(x_len)
    idxs = torch.arange(max_len, dtype=torch.long).to(x_len.device)
    padding_mask = idxs.unsqueeze(0) >= x_len.unsqueeze(1)
    return padding_mask


@torch.no_grad()
def padding_to_len(padding_mask: torch.BoolTensor) -> torch.LongTensor:
    x_len = (~padding_mask).long().sum(-1)
    return x_len


@torch.no_grad()
def update_padding_mask(
    padding_mask: torch.BoolTensor, new_len: int
) -> torch.BoolTensor:
    extra = padding_mask.shape[1] % new_len
    if extra > 0:
        padding_mask = padding_mask[:, :-extra]
    padding_mask = padding_mask.view(padding_mask.shape[0], new_len, -1)
    padding_mask = padding_mask.all(-1)
    return padding_mask


@torch.no_grad()
def add_front_padding_mask(
    padding_mask: torch.BoolTensor, pad_front_lens: torch.LongTensor
) -> None:
    for i in range(len(padding_mask)):
        if pad_front_lens[i] > 0:
            padding_mask[i, : pad_front_lens[i]] = True
