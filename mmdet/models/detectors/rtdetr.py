# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType
from ..layers import (CdnQueryGenerator, RTDETRHybridEncoder,
                      RTDETRTransformerDecoder)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dino import DINO


@MODELS.register_module()
class RTDETR(DINO):
    r"""Implementation of `DETRs Beat YOLOs on Real-time Object Detection
    <https://arxiv.org/abs/2304.08069>`_

    Code is modified from the `official github repo
    <https://github.com/lyuwenyu/RT-DETR>`_.

    Args:
        bbox_head (:obj:`ConfigDict` or dict): Config for the bounding box
            head module.
        num_group (int): Number of groups to speed detr training. Default is 1.
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator.
    """

    def __init__(self,
                 *args,
                 bbox_head: ConfigType,
                 num_group: int = 1,
                 dn_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            *args,
            bbox_head=bbox_head,
            num_group=num_group,
            dn_cfg=dn_cfg,
            **kwargs)
        self.num_group = num_group

        self.dn_query_generators = nn.ModuleList([self.dn_query_generator])
        self.dn_query_generators.extend(
            CdnQueryGenerator(**dn_cfg) for _ in range(num_group - 1))

        bbox_head['num_pred_layer'] += num_group - 1
        self.bbox_head = MODELS.build(bbox_head)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = RTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict_list = self.forward_transformer(
            img_feats, batch_data_samples)
        losses = defaultdict(int)
        for group_idx in range(self.num_group):
            group_losses = self.bbox_head.loss(
                **head_inputs_dict_list[group_idx],
                batch_data_samples=batch_data_samples)
            for name, loss in group_losses.items():
                losses[name] = losses[name] + loss / self.num_group
        return losses

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Union[List[Dict], Dict]:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        head_inputs_dict_list = []
        for group_idx in range(self.num_group):
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                group_idx,
                **encoder_outputs_dict,
                batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)

            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            if not self.training:
                return head_inputs_dict

            head_inputs_dict_list.append(head_inputs_dict)

        return head_inputs_dict_list

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'mlvl_feats' and
              'spatial_shapes'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'spatial_shapes' and
              `level_start_index`.
        """
        spatial_shapes = []
        for feat in mlvl_feats:
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            spatial_shapes.append(spatial_shape)

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))

        encoder_inputs_dict = dict(
            mlvl_feats=mlvl_feats, spatial_shapes=spatial_shapes)
        decoder_inputs_dict = dict(
            memory_mask=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=None)
        return encoder_inputs_dict, decoder_inputs_dict

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
            `memory` and `spatial_shapes`.
        """
        mlvl_feats = self.encoder(mlvl_feats)

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        # (bs, num_feat_points, dim)
        memory = torch.cat(feat_flatten, 1)

        encoder_outputs_dict = dict(
            memory=memory, memory_mask=None, spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        group_idx: int,
        memory: Tensor,
        memory_mask: Tensor,
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
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers
                                                       +
                                                       group_idx].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers
                                                        + group_idx](
                                                            output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers +
            group_idx](output_memory) + output_proposals

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
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generators[group_idx](batch_data_samples)
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
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
