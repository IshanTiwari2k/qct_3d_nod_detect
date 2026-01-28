from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from qct_3d_nod_detect.rpn import RPN3D, StandardRPNHead3d, DefaultAnchorGenerator3D, Matcher
from qct_3d_nod_detect.roi import FasterRCNNOutputLayers3D
from qct_3d_nod_detect.layers import ROIPooler3D
from qct_3d_nod_detect.roi import Box3DTransform, ROIHeads3D, FastRCNNConvFCHead3D
from qct_3d_nod_detect.backbone import build_vit_backbone_with_fpn
from qct_3d_nod_detect.layers import ShapeSpec
import math
import lightning.pytorch as pl
from torch import nn
from qct_3d_nod_detect.structures import Instances3D, Boxes3D, ImagesList3D
import lightning.pytorch as pl
import torch
from typing import Optional, List, Dict
from qct_3d_nod_detect.lightning import FasterRCNN3DLightning

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

# Dataset
import torch
from qct_3d_nod_detect.data import DetDatamoduleConfig, DetDataModule

config = DetDatamoduleConfig(
    patch_size=(128, 128, 128),
    centers_to_exclude=['dedomena',
                        'deeplesion',
                        'dedomena_non_cancer',
                        'segmed_pm',
                        'qure_internal',
                        ], # Keep only lidc

    windows=[{"wl": 0, "ww": 2000}],
    scan_annot_type="full_annot",
    cache_root_path="/raid13/data_cache/qct_jan26_master",
    annotated_by="merged_annot",
    batch_size=3,
    shuffle_train=True,
)

datamodule = DetDataModule(
    cfg=config,
)

anchor_generator_3d = DefaultAnchorGenerator3D(
    sizes=[[8], [16], [32], [64]],
    aspect_ratios_3d=[[(1.0, 1.0)], [(1.0, 1.0)], [(1.0, 1.0)], [(1.0, 1.0)]],
    strides=[4, 8, 16, 32],
    offset=0.5,
)

backbone_fpn = build_vit_backbone_with_fpn(
    variant="L",
    ckpt_path="/raid13/temp_checkpoint/best.ckpt",
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
    thresholds=[0.1, 0.2],
    labels=[0, -1, 1],
    allow_low_quality_matches=True,
)

roi_matcher = Matcher(
    thresholds=[0.1],
    labels=[0, 1],
    allow_low_quality_matches=True,
)

roi_pooler = ROIPooler3D(
    output_size=(7, 7, 7),
    canonical_level=4,
    canonical_box_size=64,
    pooler_type="ROIALign3DV2",
    scales=[2, 1, 0.5, 0.25]
)

rpn = RPN3D(
    in_features=["p2", "p3", "p4", "p5"],
    head=rpn_head_3d,
    anchor_generator=anchor_generator_3d,
    anchor_matcher=rpn_matcher,
    box3d_transform=box3d2box3d_transform,
    batch_size_per_image=128,
    positive_fraction=0.5,
    pre_nms_topk=(300, 150),
    post_nms_topk=(150, 150),
    nms_thresh=0.1,
    min_box_size=2.0,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.5,
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
    batch_size_per_image=128,
    positive_fraction=0.5,
    proposal_matcher=roi_matcher,
    roi_pooler=roi_pooler,
    proposal_append_gt=True,
    box_head=box_head,
    box_predictor=output_layers)

task_init_kwargs = dict(
    project_name="qct_nodule_detection_ishan",
    task_name="ViTDet_v0_27_01_2025",
    reuse_last_task_id=True,
    auto_connect_frameworks={"pytorch": False, "tensorboard": True},
)

from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    dirpath="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/ViTDet_v0_27_01_2025",
    filename="{epoch:02d}",
    save_last=True,
    save_top_k=-1,     # save ALL epochs
    every_n_epochs=1,
)

best_map_ckpt = ModelCheckpoint(
    dirpath="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/ViTDet_v0_27_01_2025",
    filename="best-map{epoch:02d}-mAP{val/mAP:.4f}",
    monitor="val/mAP",
    mode="max",
    save_top_k=1,
    save_last=False,
)

task = Task.init(**task_init_kwargs)

tb_logger = TensorBoardLogger(
        save_dir="/home/users/ishan.tiwari/Ishan_Nodseg/logs",
        name="ViTDet_v0_27_01_2025",
        version="default",
    )

model = GeneralizedRCNN3D(backbone=backbone_fpn, rpn=rpn, roi_heads=roi_head)
lit_model = FasterRCNN3DLightning(model=model, learning_rate=3e-4)

trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu",
    devices="auto",                   # GPU 3 or [4] for GPU 4
    precision="16-mixed",                 # Changed from int 32 to string "32"
    logger=tb_logger,
    log_every_n_steps=1,
    num_sanity_val_steps=0,
    enable_checkpointing=True,
    enable_model_summary=True,
    enable_progress_bar=True,       # Explicitly enable progress bar
    callbacks=[checkpoint_cb, best_map_ckpt],
    # gradient_clip_algorithm="norm",
    # gradient_clip_val=1.0,
    strategy=pl.strategies.DDPStrategy(
        find_unused_parameters=True,
        gradient_as_bucket_view=False
    ),
    # strategy="auto"
)

trainer.fit(
    lit_model,
    datamodule=datamodule,
    # ckpt_path="/home/users/ishan.tiwari/Ishan_Nodseg/checkpoints/toy_fasterrcnn/last.ckpt"
)