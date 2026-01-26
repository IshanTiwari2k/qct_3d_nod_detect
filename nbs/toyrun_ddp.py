from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from qct_3d_nod_detect.rpn import RPN3D, StandardRPNHead3d, DefaultAnchorGenerator3D
from qct_3d_nod_detect.roi import FasterRCNNOutputLayers3D
from qct_3d_nod_detect.layers import ROIPooler3D
from qct_3d_nod_detect.box_regression import Box3DTransform
from qct_3d_nod_detect.matcher import Matcher
from qct_3d_nod_detect.roi_heads import ROIHeads3D
from qct_3d_nod_detect.backbones import build_vit_backbone_with_fpn
from qct_3d_nod_detect.box_heads import FastRCNNConvFCHead3D
from qct_3d_nod_detect.layers import ShapeSpec
import math
import lightning.pytorch as pl
from torch import nn
from qct_3d_nod_detect.structures import Instances3D, Boxes3D, ImagesList3D
import lightning.pytorch as pl
import torch
from typing import Optional, List, Dict
from qct_3d_nod_detect.base_lightning import BaseLightningModule

class GeneralizedRCNN3D(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            rpn: nn.Module,
            roi_heads: nn.Module,
    ):

        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward_train(
            self,
            images: torch.Tensor,
            targets: Optional[List[Instances3D]] = None,
    ) -> Dict[str, torch.Tensor]:
        
        """
        Returns:
            training: dict of losses
            inference: List[Instances3D]
        """

        features: Dict[str, torch.Tensor] = self.backbone(images)

        image_list = ImagesList3D(
                tensor=images,
                image_sizes=[images.shape[-3:]] * images.shape[0],
                )
            
        proposals, rpn_losses = self.rpn(
            images=image_list,
            features=features,
            gt_instances=targets,
            training=True,
        )

        roi_losses = self.roi_heads(
            features=features,
            proposals=proposals,
            targets=targets,
            training=True
        )

        return {
            **rpn_losses,
            **roi_losses,
        }
    
    @torch.no_grad()
    def forward_inference(
        self,
        images: torch.Tensor,
    ) -> List[Instances3D]:
        
        features = self.backbone(images)

        image_list = ImagesList3D(
                tensor=images,
                image_sizes=[images.shape[-3:]] * images.shape[0],
        )

        proposals, _ = self.rpn(
            images=image_list,
            features=features,
            gt_instances=None,
            training=False,
        )

        for i, p in enumerate(proposals):
            print(f"Image {i}: #RPN proposals = {len(p)}")

        detections = self.roi_heads(
            features=features,
            proposals=proposals,
            targets=None,
            training=False
        )

        return detections

def build_targets(batch):
    targets = []

    B = len(batch["gt_boxes"])
    image_size = batch["image"].shape[-3:]

    for i in range(B):
        inst = Instances3D(image_size=image_size)
        inst.gt_boxes = Boxes3D(batch["gt_boxes"][i])
        inst.gt_classes = batch["gt_classes"][i].long()
        targets.append(inst)

    return targets

def build_image_list_3d(images: torch.Tensor) -> ImagesList3D:
    """
    Args:
        images: Tensor[B, C, D, H, W]
    """
    image_sizes = [tuple(images.shape[-3:]) for _ in range(images.shape[0])]
    return ImagesList3D(image_sizes)

def build_instances_3d(batch):
    instances = []

    for boxes, classes in zip(batch["gt_boxes"], batch["gt_classes"]):
        inst = Instances3D(image_size=batch["image"].shape[-3:])
        inst.gt_boxes = Boxes3D(boxes)
        inst.gt_classes = classes
        instances.append(inst)

    return instances

class FasterRCNN3DLightning(BaseLightningModule):

    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,
            grad_clip_val: float = 0.0,
            grad_clip_algorithm: str = "norm",
            log_on: str = "step"
    ):

        super().__init__(learning_rate=learning_rate)

        self.model = model
        self.grad_clip_val = grad_clip_val
        self.grad_clip_algorithm = grad_clip_algorithm
        self.log_on = log_on
        
    def _build_targets(
        self,
        batch
    ):
        
        targets = []
        image_size = batch['image'].shape[-3:] # (D, H, W)

        for gt_boxes, gt_classes in zip(
            batch['gt_boxes'], batch['gt_classes']
        ):
            
            centers = gt_boxes[:, :3]      # (cx, cy, cz)
            sizes = gt_boxes[:, 3:]        # (dx, dy, dz)

            half_sizes = sizes * 0.5

            boxes_cc = torch.cat(
                [
                    centers - half_sizes,  # (x1, y1, z1)
                    centers + half_sizes,  # (x2, y2, z2)
                ],
                dim=1,
            )
            
            inst = Instances3D(image_size=image_size)
            inst.gt_boxes = Boxes3D(boxes_cc)
            inst.gt_classes = gt_classes.long()
            targets.append(inst)

        return targets
    
    def forward(
            self,
            images,
    ):
        return self.model.forward_inference(images)
    
    def training_step(
            self, 
            batch, 
            batch_idx
        ):

        images = batch["image"]
        targets = self._build_targets(batch)

        loss_dict = self.model.forward_train(images, targets)
        total_loss = sum(loss_dict.values())

        self.log_dict_helper(loss_dict, prefix="train/")
        self.log("train/loss_total", total_loss, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(
            self, 
            batch, 
            batch_idx
        ):

        images = batch["image"]
        detections = self.model.forward_inference(images)

        num_boxes = sum(len(d) for d in detections)

        self.log(
            "val/num_boxes",
            num_boxes,
            prog_bar=True,
            sync_dist=True,
        )

        return detections

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        return optimizer


# Dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch

class ToySphereDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root = Path(root_dir) / split
        self.vol_dir = self.root / "volumes"
        self.tgt_dir = self.root / "targets"

        self.ids = sorted(p.stem for p in self.vol_dir.glob("*.pt"))[:100]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        volume = torch.load(self.vol_dir / f"{sid}.pt")   # (1, D, H, W)
        target = torch.load(self.tgt_dir / f"{sid}.pt")

        return {
            "image": volume,                  # Tensor[1, D, H, W]
            "gt_boxes": target["boxes"],      # Tensor[N, 6]
            "gt_classes": target["labels"],   # Tensor[N]
        }
    
class ToySphereDetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # Called once per process
        self.train_dataset = ToySphereDetectionDataset(
            root_dir=self.root_dir,
            split="train",
        )

        self.val_dataset = ToySphereDetectionDataset(
            root_dir=self.root_dir,
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate,
        )

def detection_collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_classes": [b["gt_classes"] for b in batch],
    }

datamodule = ToySphereDetectionDataModule(root_dir='/home/users/ishan.tiwari/Ishan_Nodseg/qct_3d_nod_detect/toy_dataset',
                                          batch_size=1,
                                          num_workers=3,
                                          pin_memory=False)

anchor_generator_3d = DefaultAnchorGenerator3D(
    sizes=[[8], [16], [32], [64]],
    aspect_ratios_3d=[[(1.0, 1.0)], [(1.0, 1.0)], [(1.0, 1.0)], [(1.0, 1.0)]],
    strides=[4, 8, 16, 32],
    offset=0.5,
)

backbone_fpn = build_vit_backbone_with_fpn(
    variant="L",
    ckpt_path=None,
    scales=[1, 2, 0.5, 0.25],
    out_channels=256
)

box3d2box3d_transform = Box3DTransform(
    weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    scale_clamp=math.log(1000.0),
)

rpn_head_3d = StandardRPNHead3d(
    in_channels=256,
    num_anchors=anchor_generator_3d.num_cell_anchors[0],
    box_dim=6
)

rpn_matcher = Matcher(
    thresholds=[0.1, 0.3],
    labels=[0, -1, 1],
    allow_low_quality_matches=True,
)

roi_matcher = Matcher(
    thresholds=[0.5],
    labels=[0, 1],
    allow_low_quality_matches=False,
)

roi_pooler = ROIPooler3D(
    output_size=(7, 7, 7),
    canonical_level=4,
    canonical_box_size=64,
    pooler_type="ROIALign3DV2",
    scales=[1, 2, 0.5, 0.25]
)

rpn = RPN3D(
    in_features=["p2", "p3", "p4", "p5"],
    head=rpn_head_3d,
    anchor_generator=anchor_generator_3d,
    anchor_matcher=rpn_matcher,
    box3d_transform=box3d2box3d_transform,
    batch_size_per_image=300,
    positive_fraction=0.3,
    pre_nms_topk=(1000, 500),
    post_nms_topk=(600, 600),
    nms_thresh=0.1,
    min_box_size=2.0,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
)

box_head = FastRCNNConvFCHead3D(
    input_shape=ShapeSpec(256, 7, 7, 7),
    conv_dims=[256,256],
    fc_dims=[512]
)

output_layers = FasterRCNNOutputLayers3D(
    input_dim=512,
    num_classes=1,
    box2box_transform=box3d2box3d_transform,
    cls_agnostic_bbox_reg=False,
)

roi_head = ROIHeads3D(
    num_classes=1,
    batch_size_per_image=300,
    positive_fraction=0.3,
    proposal_matcher=roi_matcher,
    roi_pooler=roi_pooler,
    proposal_append_gt=False,
    box_head=box_head,
    box_predictor=output_layers)

task_init_kwargs = dict(
    project_name="qct_nodule_detection_ishan",
    task_name="toy_run_fasterrcnn",
    reuse_last_task_id=True,
    auto_connect_frameworks={"pytorch": False, "tensorboard": True},
)

from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    dirpath="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/toy_fasterrcnn",
    filename="epoch{epoch:02d}",
    save_last=True,
    save_top_k=-1,     # save ALL epochs
    every_n_epochs=1,
)

task = Task.init(**task_init_kwargs)

tb_logger = TensorBoardLogger(
        save_dir="/home/users/ishan.tiwari/Ishan_Nodseg/logs",
        name="toy_run_fasterrcnn",
        version="default",
    )

model = GeneralizedRCNN3D(backbone=backbone_fpn, rpn=rpn, roi_heads=roi_head)
lit_model = FasterRCNN3DLightning(model=model, learning_rate=0.001)

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu",
    devices="auto",                   # GPU 3 or [4] for GPU 4
    precision="32",                 # Changed from int 32 to string "32"
    logger=tb_logger,
    log_every_n_steps=2,
    enable_checkpointing=True,
    enable_model_summary=True,
    enable_progress_bar=True,       # Explicitly enable progress bar
    callbacks=[checkpoint_cb],
    strategy=pl.strategies.DDPStrategy(
        gradient_as_bucket_view=False
    ),
)

trainer.fit(
    lit_model,
    datamodule=datamodule,
    # ckpt_path="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/toy_fasterrcnn/last.ckpt"
)
