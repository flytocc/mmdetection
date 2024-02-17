# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from ..layers import DeqRTDETRTransformerDecoder, RTDETRHybridEncoder
from .rtdetr import RTDETR


@MODELS.register_module()
class DeepEquilibriumRTDETR(RTDETR):

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = DeqRTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

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
        losses = super().loss(batch_inputs, batch_data_samples)

        if not self.decoder.grad_accumulation:
            return losses

        losses_name = set()
        for i in range(self.decoder.extra_supervisions_on_init_head + 1):
            losses_name.update(
                ('enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou',
                 f'd{i}.loss_cls', f'd{i}.dn_loss_cls', f'd{i}.loss_bbox',
                 f'd{i}.dn_loss_bbox', f'd{i}.loss_iou', f'd{i}.dn_loss_iou'))

        refine_loss = 0.0
        for name, loss in losses.items():
            if name not in losses_name and 'loss' in name:
                refine_loss = refine_loss + loss
                losses[name] = loss.detach()
        refine_loss.backward()

        return losses
