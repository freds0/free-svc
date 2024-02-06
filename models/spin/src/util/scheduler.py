import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def get_lr(optimizer: Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def noam_scheduler(
    optimizer: Optimizer, warmup: int = 4000, last_epoch: int = -1
) -> _LRScheduler:
    def func(step: int):
        if step < warmup:
            return (step + 1) / warmup
        else:
            return (warmup / (step + 1)) ** 0.5

    return LambdaLR(optimizer, func, last_epoch)


def linear_warmup_decay_scheduler(
    optimizer: Optimizer,
    warmup: int = 4000,
    max_step: int = 1000000,
    init_lr: float = 1e-6,
    final_lr: float = 1e-6,
) -> _LRScheduler:
    func_list = []

    for param_group in optimizer.param_groups:
        base_lr = param_group["lr"]
        rate_i = init_lr / base_lr
        rate_f = final_lr / base_lr

        def func(step: int) -> float:
            if step <= warmup:
                return rate_i + (1.0 - rate_i) * step / warmup
            else:
                return 1.0 - (1.0 - rate_f) * (step - warmup) / (max_step - warmup - 1)

        func_list.append(func)

    return LambdaLR(optimizer, func_list)


def linear_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup: int = 4000,
    max_step: int = 1000000,
    final_lr: float = 1e-6,
) -> _LRScheduler:
    func_list = []

    for param_group in optimizer.param_groups:
        base_lr = param_group["lr"]
        rate = final_lr / base_lr

        def func(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            else:
                q = 0.5 * (
                    1 + math.cos(math.pi * (step + 1 - warmup) / (max_step - warmup))
                )
                return (1.0 - rate) * q + rate

        func_list.append(func)

    return LambdaLR(optimizer, func_list)


def get_scheduler(name: str, optimizer: Optimizer, **kwargs) -> _LRScheduler:
    if name == "noam":
        return noam_scheduler(optimizer, **kwargs)
    elif name == "linear_warmup_decay":
        return linear_warmup_decay_scheduler(optimizer, **kwargs)
    elif name == "linear_warmup_cosine":
        return linear_warmup_cosine_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"Unknown lr scheduler {name}")
