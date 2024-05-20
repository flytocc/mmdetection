# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
from torch import Tensor

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from ..dense_heads.rtmdet_ins_head import MaskFeatModule
from .rtdetr import RTDETR


@MODELS.register_module()
class RTDETRIns(RTDETR):
    """RTDETR for Instance."""

    def __init__(self, *args, mask_stacked_convs: int = 4, **kwargs) -> None:
        self.mask_stacked_convs = mask_stacked_convs
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super()._init_layers()
        self.mask_features = MaskFeatModule(
            in_channels=self.encoder.out_channels,
            feat_channels=self.encoder.out_channels,
            stacked_convs=self.mask_stacked_convs,
            num_levels=len(self.encoder.in_channels),
            num_prototypes=self.encoder.out_channels)

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        spatial_shapes: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            dict: The output of the Transformer encoder, which includes
            `memory`,  `mask_features` and `spatial_shapes`.
        """
        mlvl_feats = self.encoder(mlvl_feats)
        mask_features = self.mask_features(mlvl_feats)

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        # (bs, num_feat_points, dim)
        memory = torch.cat(feat_flatten, 1)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=None,
            mask_features=mask_features,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        mask_features: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            mask_features (Tensor): instance mask features that
                has shape (bs, dim, h, w).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              `mask_features` and `dn_meta` when `self.training` is `True`,
              else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))

        enc_outputs_mask = self.bbox_head.mask_branches[
            self.decoder.num_layers](query)  # shuold norm?
        topk_mask = torch.einsum('bqc,bchw->bqhw', enc_outputs_mask,
                                 mask_features)

        # unified reference points
        h, w = topk_mask.shape[-2:]
        factor = topk_mask.new_tensor([w, h, w, h]).unsqueeze(0)
        masks = topk_mask.detach().reshape(-1, h, w) > 0
        topk_coords_xyxy = masks_to_boxes(masks).reshape(bs, -1, 4)
        topk_coords_normalized = bbox_xyxy_to_cxcywh(topk_coords_xyxy) / factor
        topk_coords_unact = inverse_sigmoid(topk_coords_normalized)

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()  # detach() is not used in DINO
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            enc_outputs_mask=topk_mask,
            mask_features=mask_features,
            dn_meta=dn_meta) if self.training else dict(
                mask_features=mask_features)
        return decoder_inputs_dict, head_inputs_dict


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Compute the bounding boxes around the provided masks.

    Args:
        The masks should be in format [N, H, W] where N is the number of masks,
        (H, W) are the spatial dimensions.

    Returns:
        Tensor: shape [N, 4], with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
