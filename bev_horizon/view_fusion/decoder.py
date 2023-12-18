# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.core.nus_box3d_utils import adjust_coords, get_min_max_coords
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["BevSegDecoder", "BevDetDecoder"]


logger = logging.getLogger(__name__)


class BevDecoder(nn.Module):
    """The basic decoder structure of bev.

    Args:
        head: Head module.
        name: Name for task.
        task_feat_index:
            Index for the task.
        task_weight:
            Task weight for the task.
        target:
            Target module for the task.
        decoder:
            Decoder module for the task.
        bev_size:
            Bev feature size.
        task_size:
            Task feature size.
        grid_quant_scale:
            Quant scale for grid_sample
    """

    def __init__(
        self,
        head: nn.Module,
        name: str,
        task_feat_index: int = 0,
        task_weight: float = 1.0,
        target: nn.Module = None,
        decoder: nn.Module = None,
        bev_size: Tuple[float] = None,
        task_size: Tuple[float] = None,
        grid_quant_scale: float = 1 / 512,
    ):
        super(BevDecoder, self).__init__()
        self.head = head
        self.target = target
        self.decoder = decoder
        self.name = name
        self.task_weight = task_weight
        self.bev_size = bev_size
        self.task_size = task_size
        self.task_feat_index = task_feat_index
        if (
            self.bev_size
            and self.task_size
            and self.task_size != self.bev_size
        ):
            self.grid_sample = hnn.GridSample(
                mode="bilinear",
                padding_mode="zeros",
            )
            self.quant_stub = QuantStub(grid_quant_scale)
            self.new_coords = nn.Parameter(
                self._gen_coords(), requires_grad=False
            )

    def _gen_coords(self):
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )
        bev_W = int(self.bev_size[1] * 2 / self.bev_size[2])
        bev_H = int(self.bev_size[0] * 2 / self.bev_size[2])

        task_min_x, task_max_x, task_min_y, task_max_y = get_min_max_coords(
            self.task_size
        )
        task_W = int(self.task_size[1] * 2 / self.task_size[2])
        task_H = int(self.task_size[0] * 2 / self.task_size[2])

        x = (
            np.linspace(task_min_x, task_max_x, task_W)
            .reshape((1, task_W))
            .repeat(task_H, axis=0)
        ) / bev_max_x

        y = (
            np.linspace(task_min_y, task_max_y, task_H)
            .reshape((task_H, 1))
            .repeat(task_W, axis=1)
        ) / bev_max_y

        x = (x + 1) * ((bev_W - 1) / 2)
        y = (y + 1) * ((bev_H - 1) / 2)
        new_coords = np.stack([x, y], axis=-1)
        new_coords = torch.tensor(new_coords)

        new_coords = adjust_coords(new_coords, (task_W, task_H))

        new_coords = new_coords.view(
            1, new_coords.shape[0], new_coords.shape[1], new_coords.shape[2]
        ).float()
        return new_coords

    def forward(self, feats, meta):
        feat = feats[self.task_feat_index]
        if hasattr(self, "grid_sample"):
            batch_size = feat.shape[0]

            new_coords = self.new_coords.repeat(batch_size, 1, 1, 1)
            feat = self.grid_sample(feat, self.quant_stub(new_coords))
        feat = [feat]
        pred = self.head(feat)
        return self._post_process(meta, pred)

    @fx_wrap()
    def _post_process(self, meta, pred):
        if self.training:
            gts = self._get_gts(meta)
            target = self.target(gts, pred)
            loss = self._loss(target)
            for k, v in loss.items():
                loss[k] = v * self.task_weight
            return [pred], dict(**loss)
        else:
            return self._decode(pred, meta)

    def fuse_model(self):
        if hasattr(self.head, "fuse_model"):
            self.head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self, "quant_stub"):
            self.quant_stub.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16, "saturate": True},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                    "saturate": True,
                },
            )
        if hasattr(self.head, "set_qconfig"):
            self.head.set_qconfig()


@OBJECT_REGISTRY.register
class BevSegDecoder(BevDecoder):
    """The segmentation decoder structure of bev.

    Args:
        loss: loss module.
        use_bce:
            Whether use binary cross entropy.
    """

    def __init__(
        self, loss: nn.Module = None, use_bce: bool = False, **kwargs
    ):
        super(BevSegDecoder, self).__init__(**kwargs)
        self.use_bce = use_bce
        self.loss = loss

    def _get_gts(self, meta):
        if self.use_bce:
            gts = meta["bev_seg_mask"]
        else:
            gts = meta["bev_seg_indices"]
        return gts

    def _loss(self, target):
        return self.loss(**target)

    def _decode(self, pred, meta):
        if self.decoder is not None:
            result = self.decoder(pred)
            result = {self.name: result}
            return [pred], result
        else:
            return [pred], None


@OBJECT_REGISTRY.register
class BevDetDecoder(BevDecoder):
    """The detection decoder structure of bev.

    Args:
        loss_cls: Classify loss module.
        loss_reg:
            Regression loss module
    """

    def __init__(
        self, loss_cls: nn.Module = None, loss_reg: nn.Module = None, **kwargs
    ):
        super(BevDetDecoder, self).__init__(**kwargs)
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg

    def _get_gts(self, meta):
        return meta["bev_bboxes_labels"]

    def _loss(self, targets):
        losses = {}
        for target in targets:
            task_name = target["task_name"]

            cls_target = target["cls_target"]
            cls_loss = self.loss_cls(**cls_target)
            reg_target = target["reg_target"]
            reg_loss = self.loss_reg(**reg_target)

            losses[f"{task_name}_cls_loss"] = cls_loss
            losses[f"{task_name}_reg_loss"] = reg_loss
        return losses

    def _decode(self, pred, meta):
        if self.decoder is not None:
            result = self.decoder(pred, meta)
            result = {self.name: result}
            return [pred], result
        else:
            ret = []
            for task_pred in pred:
                for _, v in task_pred.items():
                    ret.append(v)
            return ret, None
