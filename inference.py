from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.core.config_store import ConfigStore
import hydra

# from omegaconf import DictConfig, OmegaConf
from GLP_model import MyModel
from evaluate import *
from config import *
from dataset import *
import wandb


cs = ConfigStore.instance()
cs.store(name="config_s", node=DepthConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DepthConfig) -> None:

    checkpoint_reference = (
        "uy9ko/GLP_depth_decoder_finetune_contrast/model-Nyu_0.2_FDA_v2:v0"
    )
    run = wandb.init(project=cfg.logger.proj_name)
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    model = MyModel.load_from_checkpoint(
        os.path.join(artifact_dir, "model.ckpt"),
        lr=cfg.model.lr,
        loss_alphas=cfg.model.loss_alphas,
        weight_decay=cfg.model.weight_decay,
        min_depth=cfg.model.min_depth,
        max_depth=cfg.model.max_depth,
    )
    data = MyDataModule(
        data_name=cfg.data.data_name,
        data_path=cfg.data.train_data_path,
        predict_data_path=cfg.data.predict_data_path,
        scale_size=cfg.data.scale_size,
        batch_size=cfg.data.batch_size,
        numworkers=cfg.data.numworkers,
    )

    trainer = Trainer(
        default_root_dir=cfg.trainer.trainer_path,
        gpus=cfg.trainer.gpus,
    )
    predictions = trainer.predict(model, data, return_predictions=True)
    metrics = display_metrics(predictions)


if __name__ == "__main__":
    my_app()
