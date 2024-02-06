import abc

import pytorch_lightning as pl
import yaml


class BaseModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        if isinstance(config, str) and config.split(".")[-1] in {"yaml", "yml"}:
            config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)

        self.config = config
        self.save_hyperparameters(config)

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
