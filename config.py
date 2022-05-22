from dataclasses import dataclass
from tabnanny import check
from typing import List


@dataclass
class TrainerConfig:
    gpus: int
    trainer_path: str
    max_epochs: int
    auto_lr_find: bool


@dataclass
class ModelConfig:
    lr: float
    loss_alphas: List[float]
    weight_decay: float
    min_depth: float
    max_depth: float
    ckpt_path: str


@dataclass
class DataConfig:
    data_set: str
    train_data_path: str
    predict_data_path: str
    fda_trans: bool
    numworkers: int
    batch_size: int
    scale_size: List[int]


@dataclass
class LoggerConfig:
    proj_name: str
    run_name: str
    version: str
    wandb_savedir: str


@dataclass
class DepthConfig:
    trainer: TrainerConfig
    model: ModelConfig
    data: DataConfig
    logger: LoggerConfig
