from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from hydra.core.config_store import ConfigStore
import hydra
from densemodel import MyModel
from evaluate import *
from config import *
from dataset import *
import pytorch_lightning as pl
import wandb

cs = ConfigStore.instance()
cs.store(name="config_s", node=DepthConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DepthConfig) -> None:
    pl.seed_everything(seed=42)
    load_with_wandb = False
    if load_with_wandb:
        run = wandb.init(project=cfg.logger.proj_name)
        artifact = run.use_artifact(cfg.model.ckpt_path, type="model")
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, "model.ckpt")
    else:
        ckpt_path = cfg.model.ckpt_path

    wandb_logger = WandbLogger(
        project=cfg.logger.proj_name,
        log_model=True,
        version=cfg.logger.version,
        save_dir=cfg.logger.wandb_savedir,
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            every_n_train_steps=0,
            every_n_epochs=1,
            train_time_interval=None,
            save_on_train_epoch_end=None,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    model = MyModel(
        lr=cfg.model.lr,
        loss_alphas=cfg.model.loss_alphas,
        weight_decay=cfg.model.weight_decay,
        min_depth=cfg.model.min_depth,
        max_depth=cfg.model.max_depth,
        ckpt_path=ckpt_path,
    )
    data = MyDataModule(
        data_name=cfg.data.data_name,
        data_path=cfg.data.train_data_path,
        predict_data_path=cfg.data.predict_data_path,
        scale_size=cfg.data.scale_size,
        FDA_trans=cfg.data.fda_trans,
        batch_size=cfg.data.batch_size,
        numworkers=cfg.data.numworkers,
    )
    trainer = Trainer(
        default_root_dir=cfg.trainer.trainer_path,
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        auto_lr_find=cfg.trainer.auto_lr_find,
        callbacks=checkpoint_callbacks,
        log_every_n_steps=2,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    my_app()
