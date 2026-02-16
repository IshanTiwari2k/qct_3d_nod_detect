from clearml import Task
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf 
from qct_3d_nod_detect.lightning import FasterRCNN3DLightning
import hydra
from hydra.utils import instantiate
from qct_3d_nod_detect.nn import build_model_from_cfg
from clearml import Task
from qct_3d_nod_detect.data import DetDataModule

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    # Clearml
    task = Task.init(
        project_name=cfg.project.name,
        task_name=cfg.project.task_name,
        auto_connect_frameworks={
            "pytorch": False,
            "tensorboard": True,
        }
        continue_last_task=True,
        task_id="16c5ea512fa94b74adc147e069653b04"
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    datamodule_cfg = instantiate(cfg.data)
    datamodule = DetDataModule(cfg=datamodule_cfg)
    model = build_model_from_cfg(cfg.model)

    lit_model = FasterRCNN3DLightning(
        model=model,
        learning_rate=cfg.optim.lr,
        # modules_to_freeze=["backbone.backbone"] # TODO make it configurable
    )

    trainer = instantiate(cfg.trainer)
    
    trainer.fit(
        lit_model, 
        datamodule=datamodule,
        ckpt_path="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/ViTDet_atlas_10_02_27/best-map-mAPval/metrics/mAP=0.0027.ckpt"
    )

if __name__ == "__main__":
    main()