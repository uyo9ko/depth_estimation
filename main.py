from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.core.config_store import ConfigStore
import hydra
# from omegaconf import DictConfig, OmegaConf
from densemodel import MyModel
from evaluate import *
from config import *
from dataset import *



cs = ConfigStore.instance()
cs.store(name="config_s", node=DepthConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DepthConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    wandb_logger = WandbLogger(project=cfg.logger.proj_name, 
                                log_model="all",
                                version=cfg.logger.version,
                               save_dir=cfg.logger.wandb_savedir)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                        mode='min', 
                                        save_top_k=3,
                                        every_n_train_steps=0, 
                                        every_n_epochs=1, 
                                        train_time_interval=None, 
                                        save_on_train_epoch_end=None)


    data = MyDataModule(
        data_name= cfg.data.data_name,
        data_path= cfg.data.train_data_path,
        predict_data_path= cfg.data.predict_data_path,
        scale_size= cfg.data.scale_size,
        batch_size= cfg.data.batch_size,
        numworkers= cfg.data.numworkers
        )
    model=MyModel(lr= cfg.model.lr,
            loss_alphas= cfg.model.loss_alphas,
            weight_decay= cfg.model.weight_decay,
            min_depth= cfg.model.min_depth,
            max_depth= cfg.model.max_depth,
        ) 
    trainer = Trainer(
        default_root_dir=cfg.trainer.trainer_path,
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        auto_lr_find=cfg.trainer.auto_lr_find,
        enable_checkpointing = True,
        callbacks=[checkpoint_callback],
        # log_every_n_steps=2
        )

    # trainer.tune(model, datamodule=data)
    # train
    trainer.fit(model, data, ckpt_path='/root/zhshen/wandb/isw_densedepth/v2/checkpoints/epoch=4-step=15144.ckpt')

    predictions = trainer.predict(model, data, return_predictions=True)
    metrics = display_metrics(predictions)
    # metrics = display_uw_metrics(predictions)
    # wandb_logger.log_metrics(metrics)
    


if __name__ == "__main__":
    my_app()

