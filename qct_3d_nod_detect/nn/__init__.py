from .FasterRCNN3d import (
    GeneralizedRCNN3D, 
    build_instances_3d, 
    build_image_list_3d, 
    build_targets
)

import math
from hydra.utils import instantiate

from qct_3d_nod_detect.rpn import (
    RPN3D,
    StandardRPNHead3d,
    DefaultAnchorGenerator3D,
    Matcher,
)
from qct_3d_nod_detect.roi import (
    ROIHeads3D,
    FasterRCNNOutputLayers3D,
    Box3DTransform,
    FastRCNNConvFCHead3D,
)
from qct_3d_nod_detect.layers import ROIPooler3D, ShapeSpec

def build_model_from_cfg(
        cfg
):
    
    backbone = instantiate(cfg.backbone)
    
    anchor_generator = DefaultAnchorGenerator3D(
        sizes=cfg.anchor_generator.sizes,
        aspect_ratios_3d=cfg.anchor_generator.aspect_ratios_3d,
        strides=cfg.anchor_generator.strides,
        offset=cfg.anchor_generator.offset,
    )

    box3d_transform_rpn = Box3DTransform(
        weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        scale_clamp=math.log(1000.0),
    )

    rpn_head = StandardRPNHead3d(
        in_channels=cfg.backbone.out_channels,
        num_anchors=anchor_generator.num_cell_anchors[0],
        box_dim=6,
    )

    rpn_matcher = Matcher(
        thresholds=cfg.rpn.matcher.thresholds,
        labels=cfg.rpn.matcher.labels,
        allow_low_quality_matches=cfg.rpn.matcher.allow_low_quality_matches,
    )

    rpn = RPN3D(
        in_features=cfg.rpn.in_features,
        head=rpn_head,
        anchor_generator=anchor_generator,
        anchor_matcher=rpn_matcher,
        box3d_transform=box3d_transform_rpn,
        batch_size_per_image=cfg.rpn.batch_size_per_image,
        positive_fraction=cfg.rpn.positive_fraction,
        pre_nms_topk=tuple(cfg.rpn.pre_nms_topk),
        post_nms_topk=tuple(cfg.rpn.post_nms_topk),
        nms_thresh=cfg.rpn.nms_thresh,
        min_box_size=cfg.rpn.min_box_size,
        box_reg_loss_type=cfg.rpn.box_reg_loss_type,
        smooth_l1_beta=cfg.rpn.smooth_l1_beta,
    )

    roi_pooler = ROIPooler3D(
        output_size=tuple(cfg.roi.pooler.output_size),
        canonical_level=cfg.roi.pooler.cannonical_level,
        canonical_box_size=cfg.roi.pooler.cannonical_box_size,
        pooler_type=cfg.roi.pooler.type,
        scales=cfg.roi.pooler.scales,
    )

    roi_matcher = Matcher(
        thresholds=cfg.roi.matcher.thresholds,
        labels=cfg.roi.matcher.labels,
        allow_low_quality_matches=cfg.roi.matcher.allow_low_quality_matches,
    )

    box_head = FastRCNNConvFCHead3D(
        input_shape=ShapeSpec(
            channels=cfg.backbone.out_channels,
            depth=cfg.roi.pooler.output_size[0],
            height=cfg.roi.pooler.output_size[1],
            width=cfg.roi.pooler.output_size[2],
        ),
        conv_dims=cfg.box_head.conv_dims,
        fc_dims=cfg.box_head.fc_dims,
    )

    box3d_transform_roi = Box3DTransform(
        weights=(10.0, 10.0, 10.0, 5.0, 5.0, 5.0),
        scale_clamp=math.log(1000.0),
    )

    box_predictor = FasterRCNNOutputLayers3D(
        input_dim=cfg.box_head.fc_dims[-1],
        num_classes=cfg.roi.num_classes,
        box2box_transform=box3d_transform_rpn,
        cls_agnostic_bbox_reg=False,
    )

    roi_heads = ROIHeads3D(
        num_classes=cfg.roi.num_classes,
        batch_size_per_image=cfg.roi.batch_size_per_image,
        positive_fraction=cfg.roi.positive_fraction,
        proposal_matcher=roi_matcher,
        roi_pooler=roi_pooler,
        proposal_append_gt=True,
        box_head=box_head,
        box_predictor=box_predictor,
    )

    model = GeneralizedRCNN3D(
        backbone=backbone,
        rpn=rpn,
        roi_heads=roi_heads
    )

    return model