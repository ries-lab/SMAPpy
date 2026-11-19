import copy
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import hydra
import omegaconf
import pytorch_lightning as pl
import structlog
import torch

import decode
from decode.neuralfitter import model
from decode.neuralfitter.data import datamodel
from decode.neuralfitter.train import setup_cfg, auto_lr
from decode.utils import param_auto, system, hardware

logger = structlog.get_logger()


def log(cfg_raw, cfg_filled):
    logger.info("DECODE", version=decode.__version__)
    logger.info(
        "Experiment", tag=cfg_raw["Meta"]["tag"], version=cfg_raw["Meta"]["version"]
    )
    logger.info("at path", path=cfg_raw["Paths"]["experiment"])
    logger.info("System", system=system.collect_system())
    logger.info("Hardware", hardware=hardware.collect_hardware())
    logger.info("Input cfg", cfg=cfg_raw)
    logger.info("Resulting cfg", cfg=cfg_filled)


def setup_system(cfg: omegaconf.DictConfig):
    # setup some compute environment settings
    if (st := cfg.Computing.multiprocessing.sharing_strategy) is not None:
        torch.multiprocessing.set_sharing_strategy(st)
        logger.info("Set multiprocessing sharing strategy", strategy=st)

    if (sm := cfg.Computing.multiprocessing.start_method) is not None:
        torch.multiprocessing.set_start_method(sm)
        logger.info("Set multiprocessing start method", method=sm)

    torch.set_num_threads(cfg.Computing.threads)
    logger.info("Set number of threads", threads=cfg.Computing.threads)


def resolve_paths(base: Path, path: str | Path) -> Path:
    """Resolve paths relative to base path."""
    if not base.is_absolute():
        raise ValueError("Base path must be absolute for resolution.")

    path = Path(path) if not isinstance(path, Path) else path
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@hydra.main(config_path="../../../config", config_name="param_run_240802", version_base="1.2")
def train(cfg: omegaconf.DictConfig):
    if cfg.Trainer.random_seed is not None:
        # workers=False since no data augmentation (dataloaders simply load)
        pl.seed_everything(cfg.Trainer.random_seed, workers=False)

    cfg_backup = copy.copy(cfg)
    auto_cfg = param_auto.AutoConfig(
        fill=False,
        fill_test=True,
        auto_scale=True,
        return_type=omegaconf.DictConfig,
        ref={},
    )
    cfg = omegaconf.OmegaConf.to_container(cfg)
    cfg = auto_cfg.parse(cfg)

    # paths resolution
    base = cfg["Paths"]["base"]
    base = Path(base) if base is not None else Path.cwd()
    cfg["Paths"] = {
        k: str(resolve_paths(base, v)) if v is not None else v
        for k, v in cfg["Paths"].items()
    }

    # setup paths and backups
    # because hydra changes cwd interpolated by set path
    path_exp = Path(os.getcwd())
    # ToDo: This can fail for overwrites or inside container
    exp_id = (
        str(path_exp.relative_to(path_exp.parents[1]))
        if len(path_exp.parents) >= 2
        else "default_experiment"
    )

    path_cfg_in = path_exp / "param_run_in.yaml"
    path_cfg_run = path_exp / "param_run.yaml"

    omegaconf.OmegaConf.save(cfg_backup, path_cfg_in)
    omegaconf.OmegaConf.save(cfg, path_cfg_run)

    # setup logging
    path_log = path_exp / "training.log"
    fh = logging.FileHandler(filename=path_log)
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    log(cfg_backup, cfg)

    setup_system(cfg)

    exp_train = setup_cfg.setup_sampler(cfg, cfg["Simulation"])
    exp_val = setup_cfg.setup_sampler(cfg, cfg["Test"])

    dm = datamodel.DataModel(
        experiment_train=exp_train,
        experiment_val=exp_val,
        path_val=path_exp / "val_cache.pkl",
        num_workers=cfg.Hardware.cpu.worker,
        batch_size=cfg.Trainer.batch_size,
        pin_memory=cfg.Computing.multiprocessing.pin_memory,
    )
    dm.prepare_data()

    proc_train = exp_train._proc
    proc_val = exp_val._proc
    backbone = setup_cfg.setup_model(cfg)
    loss_train = setup_cfg.setup_loss(cfg, cfg["Simulation"])
    loss_val = setup_cfg.setup_loss(cfg, cfg["Test"])
    eval_filter = setup_cfg.setup_eval_filter(cfg)
    matcher = setup_cfg.setup_matcher(cfg)
    evaluator = setup_cfg.setup_evaluator(cfg)
    opt, opt_specs = setup_cfg.setup_optimizer(cfg)
    lr_sched, lr_sched_specs = setup_cfg.setup_scheduler(cfg)
    m_specs = dict(
        model=backbone,
        proc_train=proc_train,
        proc_val=proc_val,
        batch_size=cfg.Trainer.batch_size,
        optimizer=opt,
        optimizer_specs=opt_specs,
        lr=cfg.Optimizer.lr,
        lr_sched=lr_sched,
        lr_sched_specs=lr_sched_specs,
        loss_train=loss_train,
        loss_val=loss_val,
        matcher=matcher,
        evaluator=evaluator,
        eval_filter=eval_filter,
        ix_tar=cfg.Trainer.frame_window // 2,
    )
    # difference between warmstart and init checkpoint:
    # init_checkpoint also overrides LR, LR scheduler, optimizer, etc.
    # and deletes the checkpoint file afterwards (meant only to continue training);
    # warmstart only copies the weights for model initialization
    path_warmstart = cfg.Paths.get("checkpoint_warmstart")
    if path_warmstart is not None:
        if cfg.Paths.checkpoint_init is not None:
            raise ValueError("Cannot use both warmstart and init checkpoint.")
        m = model.Model.load_from_checkpoint(path_warmstart, **m_specs)
    else:
        m = model.Model(**m_specs)

    callbacks = setup_cfg.setup_callbacks(cfg, path_exp)
    logger_training = setup_cfg.setup_logger(cfg, version=exp_id)

    trainer_specs = {
        "default_root_dir": cfg.Paths.experiment,
        "accelerator": "gpu" if "cuda" in cfg.Hardware.device.lightning else "cpu",
        "devices": [int(cfg.Hardware.device.lightning.lstrip("cuda:"))]
        if "cuda" in cfg.Hardware.device.lightning
        else None,
        "precision": cfg.Computing.precision,
        "reload_dataloaders_every_n_epochs": 1,
        "logger": logger_training,
        "max_epochs": cfg.Trainer.max_epochs,
        "gradient_clip_val": cfg.Trainer.gradient_clip_val,
        "gradient_clip_algorithm": cfg.Trainer.gradient_clip_algorithm,
        "callbacks": callbacks,
        "num_sanity_val_steps": 1,
        "deterministic": False, # cfg.Trainer.random_seed is not None,
    }

    trainer = pl.Trainer(**trainer_specs)

    if m.auto_lr and not cfg.Paths.checkpoint_init:
        # find best initial rate
        auto_lr.alternative_lr_finder(trainer_specs, m, dm)
        logger.info("Found best initial learning rate", lr=m._lr)

    trainer.fit(
        model=m,
        datamodule=dm,
        ckpt_path=cfg.Paths.checkpoint_init,  # resumes training if not None
    )
