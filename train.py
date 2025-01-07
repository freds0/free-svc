import argparse
import logging
import os
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.commons as commons
import utils
from data_utils import (
    FeatureAudioSpeakerLoader,
    FeatureAudioSpeakerCollate,
    BucketBatchSampler,
    DistributedSamplerWrapper,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_processing

assert torch.cuda.is_available(), "CPU training is not allowed."
torch.backends.cudnn.benchmark = True
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, config: DictConfig, run_dir, logger):
        print(OmegaConf.to_yaml(config))
        self.logger = logger
        self.config = config
        self.run_dir = run_dir
        if config.model.save_dir is None:
            self.save_dir = self.run_dir
        else:
            self.save_dir = config.model.save_dir
        self.step = 1
        self.epoch = 0
        self.n_data_loader_workers = self.config.data.num_workers
        self.scaler = GradScaler(enabled=config.train.fp16_run)

    def _train_step(self, net_g, net_d, optim_g, optim_d, c, spec, y, pitch, spk=None, lang_id=None, rank=0, writer=None, writer_valid=None):

        self.logger.debug(f"c: {c.shape if c is not None else None}, spec: {spec.shape}, y: {y.shape}, pitch: {pitch.shape}, g: {spk.shape if spk is not None else None}")
        spec = spec.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        pitch = pitch.cuda(rank, non_blocking=True)
        if c is not None:
            c = c.cuda(rank, non_blocking=True)
        mel = mel_processing.spec_to_mel_torch(
            spec,
            self.config.data.filter_length,
            self.config.data.n_mel_channels,
            self.config.data.sampling_rate,
            self.config.data.mel_fmin,
            self.config.data.mel_fmax)

        with autocast(enabled=self.config.train.fp16_run):
            y_hat, ids_slice, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                   spec=spec, y=y, c=c, g=spk, mel=mel, pitch=pitch, lang_id=lang_id
                )

            y_mel = commons.slice_segments(
                mel, ids_slice, self.config.train.segment_size // self.config.data.hop_length)
            y_hat_mel = mel_processing.mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.hop_length,
                self.config.data.win_length,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax
            )
            y = commons.slice_segments(
                y, ids_slice * self.config.data.hop_length, self.config.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        self.scaler.scale(loss_disc_all).backward()
        self.scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        self.scaler.step(optim_d)

        with autocast(enabled=self.config.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * \
                    self.config.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                z_mask) * self.config.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()

        self.scaler.scale(loss_gen_all).backward()
        self.scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        self.scaler.step(optim_g)

        self.scaler.update()

        if rank == 0:
            if self.step % self.config.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = {
                    "disc": loss_disc,
                    "gen": loss_gen,
                    "fm": loss_fm,
                    "mel": loss_mel,
                    "kl": loss_kl
                }

                info = {k: float(losses[k]) for k in losses}
                info["epoch"] = self.epoch
                info["step"] = self.step
                info["lr"] = lr
                self.logger.info(str(info))

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g
                }
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=self.step,
                    images=image_dict,
                    scalars=scalar_dict)
        self.step += 1

    def _train_one_epoch(self, nets, optims, train_loader, valid_loader=None, rank=0, writer=None, writer_valid=None):
        assert valid_loader is not None or rank != 0, "Validation loader is required for rank 0"
        self.logger.info("Start training epoch {}...".format(self.epoch))
        net_g, net_d = nets
        optim_g, optim_d = optims

        train_loader.batch_sampler.set_epoch(self.epoch)

        net_g.train()
        net_d.train()

        if rank==0:
            self.evaluate(generator=net_g, valid_loader=valid_loader, writer_valid=writer_valid)

        for batch_idx, items in tqdm(enumerate(train_loader), total=len(train_loader)):
            try:
                if self.config.data.use_spk_emb and not self.config.data.get("use_lang_emb", False):
                    c, spec, y, pitch, spk = items
                    spk = spk.cuda(rank, non_blocking=True)
                elif self.config.data.use_spk_emb and self.config.data.get("use_lang_emb", False):
                    c, spec, y, pitch, spk, lang_id = items
                    spk = spk.cuda(rank, non_blocking=True)
                    lang_id = lang_id.cuda(rank, non_blocking=True)
                elif self.config.data.get("use_lang_emb", False) and not self.config.data.use_spk_emb:
                    c, spec, y, pitch, lang_id = items
                    spk = None
                    lang_id = lang_id.cuda(rank, non_blocking=True)
                else:
                    c, spec, y, pitch = items
                    spk = None
                    lang_id = None
                self._train_step(
                    net_g=net_g,
                    net_d=net_d,
                    optim_g=optim_g,
                    optim_d=optim_d,
                    c=c,
                    spec=spec,
                    y=y,
                    pitch=pitch,
                    spk=spk,
                    lang_id=lang_id,
                    rank=rank,
                    writer=writer
                )

                if rank==0 and self.step % self.config.train.valid_steps_interval == 0:
                    self.evaluate(generator=net_g, valid_loader=valid_loader, writer_valid=writer_valid)
                if rank==0 and self.step % self.config.train.save_steps_interval == 0:
                    utils.save_checkpoint(net_g, optim_g, self.config.train.learning_rate, self.epoch, os.path.join(
                        self.save_dir, f"G_{self.epoch:05d}_{self.step:07d}.pth"))
                    utils.save_checkpoint(net_d, optim_d, self.config.train.learning_rate, self.epoch, os.path.join(
                        self.save_dir, f"D_{self.epoch:05d}_{self.step:07d}.pth"))

            except Exception as e:  # TODO: temporary here because there was some issues in the dataset
                logger.error(f"Error on step {self.step} (might indicate a problem with the dataset): {str(e)}")
                if self.config.train.raise_error:
                    raise e


    def evaluate(self, generator, valid_loader, writer_valid=None):
        self.logger.info("Evaluating...")
        generator.eval()
        with torch.no_grad():
            for batch_idx, items in tqdm(enumerate(valid_loader)):
                if self.config.data.use_spk_emb and not self.config.data.get("use_lang_emb", False):
                    c, spec, y, pitch, spk = items
                    g = spk[:1].cuda(0)
                    lang_id = None
                elif self.config.data.use_spk_emb and self.config.data.get("use_lang_emb", False):
                    c, spec, y, pitch, spk, lang_id = items
                    g = spk[:1].cuda(0)
                    lang_id = lang_id.cuda(0)
                elif self.config.data.get("use_lang_emb", False) and not self.config.data.use_spk_emb:
                    c, spec, y, pitch, lang_id = items
                    g = None
                    lang_id = lang_id.cuda(0)
                else:
                    c, spec, y, pitch = items
                    g = None
                    lang_id = None
                spec, y, pitch = spec[:1].cuda(0), y[:1].cuda(0), pitch[:1].cuda(0)
                if c is not None:
                    c = c[:1].cuda(0)

                mel = mel_processing.spec_to_mel_torch(
                    spec,
                    self.config.data.filter_length,
                    self.config.data.n_mel_channels,
                    self.config.data.sampling_rate,
                    self.config.data.mel_fmin,
                    self.config.data.mel_fmax)

                y_hat = generator.module.infer(c=c, y=y, g=g, mel=mel, pitch=pitch, lang_id=lang_id)

                y_hat_mel = mel_processing.mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    self.config.data.filter_length,
                    self.config.data.n_mel_channels,
                    self.config.data.sampling_rate,
                    self.config.data.hop_length,
                    self.config.data.win_length,
                    self.config.data.mel_fmin,
                    self.config.data.mel_fmax
                )

                # TODO: add more metrics

                if writer_valid:
                    image_dict = {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
                    }
                    audio_dict = {
                        f"gen/audio_{batch_idx}": y_hat[0],
                        f"gt/audio_{batch_idx}": y[0]
                    }
                    utils.summarize(
                        writer=writer_valid,
                        global_step=self.step,
                        images=image_dict,
                        audios=audio_dict,
                        audio_sampling_rate=self.config.data.sampling_rate
                    )
        generator.train()

    def get_dataset_samples_weight(self, dataset_attributes):
        key_names = np.array(dataset_attributes)
        attr_names_samples = key_names
        unique_attr_names = np.unique(attr_names_samples).tolist()
        attr_idx = [unique_attr_names.index(l) for l in tqdm(attr_names_samples)]
        attr_count = np.array(
            [len(np.where(attr_names_samples == l)[0]) for l in tqdm(unique_attr_names)])
        weight_attr = 1.0 / attr_count
        self.logger.debug(
            "Using weighted batch sampling with the following weights:")
        for k, w in zip(unique_attr_names, weight_attr):
            self.logger.debug(
                f"{k.ljust(max([len(s) for s in key_names]))}: {w:.4f}")

        dataset_samples_weight = np.array(
            [weight_attr[l] for l in attr_idx])
        dataset_samples_weight = dataset_samples_weight / \
            np.linalg.norm(dataset_samples_weight)
        dataset_samples_weight = torch.from_numpy(
            dataset_samples_weight).float()

        return dataset_samples_weight

    def train(self, rank=0, n_gpus=1):
        self.logger.info("Creating train dataloader")
        train_dataset = FeatureAudioSpeakerLoader(
            self.config.data.training_files, self.config)

        if self.config.train.weighted_batch_speaker_sampling or self.config.train.weighted_batch_lang_sampling:
            self.logger.info("Configuring weighted batch sampling. This may take a while...")

            if self.config.train.weighted_batch_speaker_sampling:
                dataset_samples_weight_spk = self.get_dataset_samples_weight(train_dataset.speakers)
                self.logger.debug(dataset_samples_weight_spk)
            if self.config.train.weighted_batch_lang_sampling:
                dataset_samples_weight_lang = self.get_dataset_samples_weight(train_dataset.lang)
                self.logger.debug(dataset_samples_weight_lang)

            if self.config.train.weighted_batch_speaker_sampling and self.config.train.weighted_batch_lang_sampling:
                dataset_samples_weight = (
                    dataset_samples_weight_spk*self.config.train.weighted_batch_speaker_sampling +
                    dataset_samples_weight_lang*self.config.train.weighted_batch_lang_sampling
                )
            elif self.config.train.weighted_batch_speaker_sampling:
                dataset_samples_weight = dataset_samples_weight_spk*self.config.train.weighted_batch_speaker_sampling
            elif self.config.train.weighted_batch_lang_sampling:
                dataset_samples_weight = dataset_samples_weight_lang*self.config.train.weighted_batch_lang_sampling
            w_sampler = WeightedRandomSampler(
                dataset_samples_weight, len(dataset_samples_weight))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=train_dataset,
                batch_size=self.config.train.batch_size,
                sort_key=lambda x: os.path.getsize(x[0]),
                drop_last=True)
            train_sampler = DistributedSamplerWrapper(batch_sampler,
                                                      num_replicas=n_gpus,
                                                      rank=rank,
                                                      shuffle=True)

            collate_fn = FeatureAudioSpeakerCollate(self.config, train_dataset)
            train_loader = DataLoader(train_dataset,
                                      num_workers=self.n_data_loader_workers,
                                      shuffle=False,
                                      pin_memory=True,
                                      collate_fn=collate_fn,
                                      batch_sampler=train_sampler)
        else:
            train_sampler = DistributedBucketSampler(
                train_dataset,
                self.config.train.batch_size,
                [32, 300, 400, 500, 600, 700, 800, 900, 1000],
                num_replicas=n_gpus,
                rank=rank,
                shuffle=True)
            collate_fn = FeatureAudioSpeakerCollate(self.config, train_dataset)
            train_loader = DataLoader(train_dataset,
                                      num_workers=self.config.data.num_workers,
                                      shuffle=False,
                                      pin_memory=True,
                                      collate_fn=collate_fn,
                                      batch_sampler=train_sampler)

        if rank == 0:
            self.logger.info("Creating valid dataloader")
            valid_dataset = FeatureAudioSpeakerLoader(
                self.config.data.validation_files, self.config)
            valid_loader = DataLoader(valid_dataset,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=1,
                                      pin_memory=False,
                                      drop_last=False,
                                      collate_fn=collate_fn)

        writer_train = SummaryWriter(log_dir=os.path.join(
            self.run_dir, self.config.tb_log_dir, "train"))
        writer_valid = SummaryWriter(log_dir=os.path.join(
            self.run_dir, self.config.tb_log_dir, "valid"))

        if self.config.train.distributed:
            dist.init_process_group(
                backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
            torch.manual_seed(self.config.seed)
            torch.cuda.set_device(rank)

        self.logger.info("Creating models...")

        net_g = SynthesizerTrn(
            self.config.data.filter_length // 2 + 1,
            self.config.train.segment_size // self.config.data.hop_length,
            config=self.config,
            **self.config.model,
        ).cuda(rank)
        net_d = MultiPeriodDiscriminator(
            self.config.model.use_spectral_norm).cuda(rank)
        optim_g = torch.optim.AdamW(
            net_g.parameters(),
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps)
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps)

        if self.config.train.distributed:
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])

        self.logger.info("\nDiscriminator:" + str(net_d))
        self.logger.info("\nGenerator:" + str(net_g))

        if self.config.train.resume_training:
            self.logger.info(
                f"Resuming training from checkpoint at {self.run_dir}")
            generator_path = utils.latest_checkpoint_path(
                self.run_dir, "G_*.pth")
            discriminator_path = utils.latest_checkpoint_path(
                self.run_dir, "D_*.pth")
            if generator_path is not None and discriminator_path is not None:
                _, _, _, epoch_str = utils.load_checkpoint(
                    generator_path, net_g, optim_g)
                _, _, _, epoch_str = utils.load_checkpoint(
                    discriminator_path, net_d, optim_d)
                self.step = (epoch_str - 1) * len(train_loader)
            else:
                self.logger.info(
                    "No checkpoint found. Starting from scratch...")
                epoch_str = 1
        else:
            if self.config.model.finetune_from_model.generator:
                self.logger.info(f"Finetuning from model {self.config.model.finetune_from_model.generator}")
                net_g = utils.load_weights(net_g, self.config.model.finetune_from_model.generator, strict=False).cuda(rank)
            if self.config.model.finetune_from_model.discriminator:
                self.logger.info(f"Finetuning from model {self.config.model.finetune_from_model.discriminator}")
                net_d = utils.load_weights(net_d, self.config.model.finetune_from_model.discriminator, strict=False).cuda(rank)
            epoch_str = 1

        self.epoch = int(epoch_str)

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.config.train.lr_decay, last_epoch=epoch_str-2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.config.train.lr_decay, last_epoch=epoch_str-2)

        self.nets = net_g, net_d
        self.optimizers = optim_g, optim_d
        self.schedulers = scheduler_g, scheduler_d

        self.logger.info("Start training")
        for epoch in range(int(self.epoch), int(self.config.train.epochs) + 1):
            self.epoch = epoch
            start_time = time.time()

            if rank == 0:
                self._train_one_epoch(rank=rank,
                                      nets=[net_g, net_d],
                                      optims=[optim_g, optim_d],
                                      train_loader=train_loader,
                                      valid_loader=valid_loader,
                                      writer=writer_train,
                                      writer_valid=writer_valid)

                if self.epoch % self.config.train.valid_epoch_interval == 0:
                    self.evaluate(generator=net_g, valid_loader=valid_loader, writer_valid=writer_valid)
                if self.epoch % self.config.train.save_epoch_interval == 0:
                    utils.save_checkpoint(net_g, optim_g, self.config.train.learning_rate, self.epoch, os.path.join(
                        self.save_dir, f"G_{self.epoch:05d}_{self.step:07d}.pth"))
                    utils.save_checkpoint(net_d, optim_d, self.config.train.learning_rate, self.epoch, os.path.join(
                        self.save_dir, f"D_{self.epoch:05d}_{self.step:07d}.pth"))

            else:
                self._train_one_epoch(rank=rank,
                                      nets=[net_g, net_d],
                                      optims=[optim_g, optim_d],
                                      train_loader=train_loader)


            if rank == 0:
                self.logger.info("End of epoch {} | Time: {:.3f}s".format(
                    self.epoch, time.time() - start_time))

            scheduler_g.step()
            scheduler_d.step()


@hydra.main(version_base=None,
            config_path="configs",
            config_name="config")
def main(cfg: DictConfig):

    run_dir = HydraConfig.get().run.dir

    logger.setLevel(cfg.log_level)
    logger.info("Log level: {}".format(cfg.log_level))

    logger.info(HydraConfig.get())

    trainer = Trainer(cfg, run_dir, logger)

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.train.port)

    if cfg.train.distributed and n_gpus > 1:
        if not cfg.train.use_multiprocessing:
            raise ValueError(
                "Distributed training is only supported in multiprocessing mode.")

    if cfg.train.use_multiprocessing:
        # TODO: add hydra logging support for mp.spawn
        logger.warning(
            "Logging is not supported in multiprocessing mode. See https://github.com/facebookresearch/hydra/issues/1126")
        mp.spawn(trainer.train, nprocs=n_gpus, args=(n_gpus,))
    else:
        trainer.train(n_gpus=n_gpus)


if __name__ == "__main__":
    main()
