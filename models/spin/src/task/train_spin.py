import argparse

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from torch.utils.data import DataLoader

from src.data import AudioPretrainPnmiValDataset, val_collate_fn
from src.model import SpinModel
from src.util import set_logging, set_pl_logger


class SpinPretrainTask:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("task", help="Task name")
        parser.add_argument("--config", "-c", help="Config .yaml file")
        parser.add_argument("--save-path", "-s", help="Path to save exp")
        parser.add_argument("--resume", "-r", default="", help="Resume training")
        parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs")
        parser.add_argument(
            "--njobs", "-j", type=int, default=8, help="Number of workers"
        )
        parser.add_argument("--seed", type=int, default=7122, help="Random seed")
        parser.add_argument("--log-level", default="info", help="Logging level")
        args = parser.parse_args()

        if not torch.cuda.is_available():
            args.device = "cpu"
            args.gpus = 0
        else:
            args.device = "cuda" if args.gpus > 0 else "cpu"

        self.args = args
        set_logging(args.log_level)

    def run(self, model_cls=SpinModel):
        assert isinstance(self.args, argparse.Namespace)

        config = yaml.load(open(self.args.config, "r"), Loader=yaml.FullLoader)
        self.config = config

        use_ddp = (
            config["trainer"].get("strategy", "").startswith("ddp")
            and self.args.gpus > 1
        )

        if self.args.save_path != "":
            config["trainer"]["default_root_dir"] = self.args.save_path

        model_checkpoint = ModelCheckpoint(
            dirpath=config["trainer"]["default_root_dir"], **config["checkpoint"]
        )

        config["trainer"]["logger"] = set_pl_logger(
            config["trainer"]["logger"],
            config["logger"]["project"],
            config["trainer"]["default_root_dir"].split("/")[-1],
        )

        trainer = Trainer(
            callbacks=[
                TQDMProgressBar(),
                model_checkpoint,
                LearningRateMonitor("step"),
            ],
            enable_progress_bar=True,
            devices=self.args.gpus,
            check_val_every_n_epoch=None,
            use_distributed_sampler=False,
            sync_batchnorm=use_ddp,
            **config["trainer"],
        )

        seed_everything(self.args.seed)

        if config.get("val_data", None) is not None:
            val_dataset = AudioPretrainPnmiValDataset(**config["val_data"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["hparam"]["val_batch_size"],
                num_workers=self.args.njobs,
                pin_memory=True,
                collate_fn=val_collate_fn,
                shuffle=False,
                drop_last=False,
            )
        else:
            val_dataset = None
            val_loader = None

        if self.args.resume != "":
            model = model_cls.load_from_checkpoint(self.args.resume)
        else:
            self.args.resume = None
            model = model_cls(config, 2)

        model.set_random_seed(self.args.seed)
        model.set_njobs(self.args.njobs)
        model.set_use_ddp(use_ddp)

        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=self.args.resume)
