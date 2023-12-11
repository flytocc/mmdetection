# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .dino_layers import DinoTransformerDecoder
from .utils import MLP, inverse_sigmoid


class CSPLayer(BaseModule):
    """CSPLayer from RTDETR.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 1.0.
        num_blocks (int): Number of blocks. Defaults to 3.
        conv_cfg (:obj:`ConfigDict`, optional): Config dict for convolution
            layer. Defaults to None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict`, optional): Config dict for normalization
            layer. Defaults to dict(type='BN', requires_grad=True)
        act_cfg (:obj:`ConfigDict`, optional): Config dict for activation
            layer. Defaults to dict(type='SiLU', inplace=True)
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 1.0,
                 num_blocks: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        block_cfg = dict(
            type='mmyolo.RepVGGBlock',
            in_channels=mid_channels,
            out_channels=mid_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_bn_first=False)
        self.blocks = nn.Sequential(*[
            MODELS.build(block_cfg) for _ in range(num_blocks)])
        if mid_channels != out_channels:
            self.final_conv = ConvModule(
                mid_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        return self.final_conv(x_main + x_short)


@MODELS.register_module()
class RTDETRTHybridEncoder(BaseModule):
    """HybridEncoder of RTDETR.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the layer.
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        out_channels (int, optional): The output dimension of the MLP.
            Defaults to 256.
        expansion (float, optional): The expansion of the CSPLayer.
            Defaults to 1.0.
        depth_mult (float, optional): The depth multiplier of the CSPLayer.
            Defaults to 1.0.
        pe_temperature (float, optional): The temperature of the positional
            encoding. Defaults to 10000.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 use_encoder_idx: List[int] = [2],
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [256, 256, 256],
                 out_channels: int = 256,
                 expansion: float = 1.0,
                 depth_mult: float = 1.0,
                 pe_temperature: float = 10000.0,
                 upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        num_csp_blocks = round(3 * depth_mult)

        # encoder transformer
        self.transformer_blocks = nn.ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))])

        # top-down fpn
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.,
        device=None,
    ) -> Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, ('Embed dimension must be divisible by 4 '
                                    'for 2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device)
        omega = temperature**(omega / -pos_dim)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w), torch.cos(out_w),
            torch.sin(out_h), torch.cos(out_h)
        ], axis=1)[None, :, :]

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        inputs = list(inputs)

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = inputs[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = inputs[enc_ind].flatten(2).permute(
                    0, 2, 1).contiguous()
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h,
                    embed_dim=self.out_channels,
                    temperature=self.pe_temperature,
                    device=src_flatten.device)
                memory = self.transformer_blocks[i](
                    src_flatten, query_pos=pos_embed, key_padding_mask=None)
                inputs[enc_ind] = memory.permute(0, 2, 1).contiguous().reshape(
                    -1, self.out_channels, h, w)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


class RTDETRTransformerDecoder(DinoTransformerDecoder):
    """Transformer decoder of RT-DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)
        self.norm = nn.Identity()  # without norm

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


def register_stash_grad_hook(v: Tensor) -> Tensor:
    leaf = v.detach().requires_grad_(True)
    def recover_stashed_grad_from_leaf(grad):
        return grad if leaf.grad is None else grad + leaf.grad
    v.register_hook(recover_stashed_grad_from_leaf)
    return leaf


class DeqRTDETRTransformerDecoder(RTDETRTransformerDecoder):
    """Transformer decoder of Deep Equilibrium RT-DETR."""

    def __init__(self,
                 *args,
                 num_layers: int,
                 refinement_steps: int,
                 supervision_position: Sequence[int],
                 grad_accumulation: bool = False,
                 perturb_query_prob: float = 0.2,
                 perturb_query_intensity: float = 0.1,
                 perturb_ref_points_prob: float = 0.2,
                 perturb_ref_points_intensity: float = 1 / 32,
                 extra_supervisions_on_init_head: int = 2,
                 rag: int = 2,
                 return_intermediate: bool = True,
                 **kwargs) -> None:
        assert num_layers == 2
        assert return_intermediate
        assert supervision_position[-1] == refinement_steps
        self.refinement_steps = refinement_steps
        self.supervision_position = supervision_position
        self.grad_accumulation = grad_accumulation
        self.perturb_query_prob = perturb_query_prob
        self.perturb_query_intensity = perturb_query_intensity
        self.perturb_ref_points_prob = perturb_ref_points_prob
        self.perturb_ref_points_intensity = perturb_ref_points_intensity
        self.extra_supervisions_on_init_head = extra_supervisions_on_init_head
        self.rag = rag
        super().__init__(num_layers, *args,
                         return_intermediate=return_intermediate, **kwargs)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[List[Tensor]]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        if not self.training:
            return self.predict(query, value, key_padding_mask, self_attn_mask,
                                reference_points, spatial_shapes,
                                level_start_index, valid_ratios, reg_branches,
                                **kwargs)

        intermediate = []
        intermediate_reference_points = []

        query, new_reference_points = self.forward_layer(
            0, query, value, key_padding_mask, self_attn_mask,
            reference_points, spatial_shapes, level_start_index,
            valid_ratios, reg_branches, **kwargs)
        reference_points = new_reference_points.detach()
        intermediate.append(query)
        intermediate_reference_points.append(new_reference_points)

        sup_query, sup_reference_points = query, reference_points
        for _ in range(self.extra_supervisions_on_init_head):
            sup_query, sup_new_reference_points = self.forward_layer(
                1, sup_query, value, key_padding_mask, self_attn_mask,
                sup_reference_points, spatial_shapes, level_start_index,
                valid_ratios, reg_branches, **kwargs)
            sup_reference_points = sup_new_reference_points.detach()
            intermediate.append(sup_query)
            intermediate_reference_points.append(sup_new_reference_points)

        if self.grad_accumulation:
            value = register_stash_grad_hook(value)

        query = query.detach()
        for i in range(1, self.refinement_steps + 1):
            with torch.no_grad():
                if random.random() < self.perturb_query_prob:
                    noise = torch.randn_like(query)
                    query = (1 - self.perturb_query_intensity) * query \
                          + self.perturb_query_intensity * noise * torch.norm(
                              query, dim=-1, keepdim=True)
                if random.random() < self.perturb_ref_points_prob:
                    noise = torch.randn_like(reference_points)
                    tmp = bbox_cxcywh_to_xyxy(reference_points) \
                        + noise * self.perturb_ref_points_intensity
                    reference_points = bbox_xyxy_to_cxcywh(tmp)
                query, new_reference_points = self.forward_layer(
                    1, query, value, key_padding_mask, self_attn_mask,
                    reference_points, spatial_shapes, level_start_index,
                    valid_ratios, reg_branches, **kwargs)
                reference_points = new_reference_points.detach()

            if i in self.supervision_position:
                sup_query, sup_reference_points = query, reference_points
                for _ in range(self.rag):
                    sup_query, sup_new_reference_points = self.forward_layer(
                        1, sup_query, value, key_padding_mask, self_attn_mask,
                        sup_reference_points, spatial_shapes,
                        level_start_index, valid_ratios, reg_branches,
                        **kwargs)
                    sup_reference_points = sup_new_reference_points.detach()
                intermediate.append(sup_query)
                intermediate_reference_points.append(sup_new_reference_points)

        return intermediate, intermediate_reference_points

    def predict(self, query: Tensor, value: Tensor,
                key_padding_mask: Tensor, self_attn_mask: Tensor,
                reference_points: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                reg_branches: nn.ModuleList, **kwargs) -> Tuple[List[Tensor]]:
        query, new_reference_points = self.forward_layer(
            0, query, value, key_padding_mask, self_attn_mask,
            reference_points, spatial_shapes, level_start_index,
            valid_ratios, reg_branches, **kwargs)
        reference_points = new_reference_points.detach()

        for i in range(1, self.refinement_steps + 1):
            query, new_reference_points = self.forward_layer(
                1, query, value, key_padding_mask, self_attn_mask,
                reference_points, spatial_shapes, level_start_index,
                valid_ratios, reg_branches, **kwargs)
            reference_points = new_reference_points.detach()

        return [query], [new_reference_points]

    def forward_layer(self, layer_idx: int, query: Tensor, value: Tensor,
                      key_padding_mask: Tensor, self_attn_mask: Tensor,
                      reference_points: Tensor, spatial_shapes: Tensor,
                      level_start_index: Tensor, valid_ratios: Tensor,
                      reg_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        reference_points_input = reference_points[:, :, None]
        query_pos = self.ref_point_head(reference_points)

        query = self.layers[layer_idx](
            query,
            query_pos=query_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            self_attn_mask=self_attn_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points_input,
            **kwargs)

        if reg_branches is not None:
            tmp = reg_branches[layer_idx](query)
            assert reference_points.shape[-1] == 4
            new_reference_points = tmp + inverse_sigmoid(
                reference_points, eps=1e-3)
            new_reference_points = new_reference_points.sigmoid()

        return query, new_reference_points
