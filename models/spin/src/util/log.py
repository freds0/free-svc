import logging
from typing import Union

from pytorch_lightning.loggers import WandbLogger


def set_logging(log_level: str = "info") -> None:
    level = getattr(logging, str(log_level).upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(filename)s.%(funcName)s %(message)s",
        datefmt="%m-%d %H:%M",
    )


def set_pl_logger(
    logger_type: Union[bool, str],
    project: str = "speech_disentangle",
    name: str = "example",
):
    if isinstance(logger_type, bool):
        return logger_type
    elif logger_type == "wandb":
        logger = WandbLogger(project=project, name=name)
        return logger
    else:
        raise NotImplementedError(f"Unknown logger type = {logger_type}")
