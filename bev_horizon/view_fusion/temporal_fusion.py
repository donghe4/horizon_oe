# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from hbdk.torch_script.placeholder import placeholder
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qat_mode import tricks
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.core.nus_box3d_utils import get_min_max_coords
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)


class TemporalFusion(nn.Module):
    """Temporal fusion for bev feats.

    Args:
        in_channels: Channels for input.
        out_channels: Channels for ouput.
        num_seq: Number of sequence for multi frames.
        bev_size: Bev size.
        grid_size: Grid size.
        num_encoder: Number of encoder layers.
        num_project: Number of project layers.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.
        use_cache_history: Use history prev feat for align bpu.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_seq: int,
        bev_size: Tuple[float],
        grid_size: Tuple[float],
        num_encoder: int = 2,
        num_project: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scale: float = 1 / 512,
        use_cache_history: bool = False,
    ):
        super(TemporalFusion, self).__init__()
        self.num_seq = num_seq
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.use_cache_history = use_cache_history
        self.in_channels = in_channels

        if self.use_cache_history is True:
            cache_tensor = torch.zeros(
                (
                    1,
                    in_channels,
                    int((bev_size[0] * 2) / bev_size[2]),
                    int((bev_size[1] * 2) / bev_size[2]),
                )
            )
            self.cache_history = nn.Parameter(
                cache_tensor, requires_grad=False
            )

        encoder = nn.ModuleList()
        for i in range(num_encoder):
            encoder.append(
                SeparableConvModule2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.encoder = nn.Sequential(*encoder)

        project = nn.ModuleList()
        for i in range(num_project):
            project.append(
                SeparableConvModule2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(*project)

        self.coords = nn.Parameter(self._gen_coords(), requires_grad=False)

        self.offset = nn.Parameter(self._gen_offset(), requires_grad=False)
        self.grid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )
        self.quant_stub = QuantStub(grid_quant_scale)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _set_scale(self, feat):
        if self.training and isinstance(feat, QTensor):
            self.quant.activation_post_process.scale = feat.scale
        return feat

    def forward(self, feats, meta, compile_model, **kwargs):
        n, c, h, w = feats.shape
        feats = self.encoder(feats)
        feat = self._fusion(feats, meta, compile_model)
        feat = self._set_scale(feat)
        prev_feat = self.dequant(feat)
        if self.use_cache_history is True:
            feat = self._cache(feat, prev_feat)
        return feat, prev_feat

    def _gen_offset(self):
        W = self.grid_size[0]
        H = self.grid_size[1]

        bev_x = (
            torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)
        ).double()
        bev_y = (
            torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)
        ).double()

        bev_offset = torch.stack([bev_x, bev_y], axis=-1) * -1
        bev_offset = bev_offset.unsqueeze(0)
        return bev_offset

    def _gen_coords(self):
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W))
            .repeat(H, 1)
        ).double()
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1))
            .repeat(1, W)
        ).double()

        coords = torch.stack([x, y], dim=-1).unsqueeze(0)

        return coords

    def _get_matrix(self, meta, idx, bev_x, bev_y):
        ego2global = meta["ego2global"]
        ego2global = np.array(ego2global)
        prev_e2g = ego2global[:, idx + 1]
        prev_g2e = np.linalg.inv(prev_e2g)
        cur_e2g = ego2global[:, idx]
        wrap_m = prev_g2e @ cur_e2g
        wrap_r = wrap_m[:, :2, :2].transpose((0, 2, 1))
        wrap_t = wrap_m[:, :2, 3]
        wrap_t = wrap_t + np.array([bev_x, bev_y])

        wrap_r /= self.bev_size[2]
        wrap_t /= self.bev_size[2]
        return wrap_r, wrap_t

    @fx_wrap()
    def _get_reference_points(self, feat, meta, idx):
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        wrap_r, wrap_t = self._get_matrix(meta, idx, bev_max_x, bev_max_y)
        wrap_r = torch.tensor(wrap_r).to(device=feat.device)
        wrap_t = torch.tensor(wrap_t).to(device=feat.device)

        new_coords = []
        batch = wrap_r.shape[0]
        for i in range(batch):
            new_coord = torch.matmul(self.coords, wrap_r[i].double()).float()
            new_coord += wrap_t[i]
            new_coord += self.offset
            new_coords.append(new_coord)
        new_coords = torch.cat(new_coords)

        return new_coords

    def _transform(self, feat, points):
        feat = self.grid_sample(
            feat,
            self.quant_stub(points),
        )
        return feat

    def fuse_model(self):
        for mod_list in [self.encoder, self.project]:
            for mod in mod_list:
                if hasattr(mod, "fuse_model"):
                    mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        tricks.fx_force_duplicate_shared_convbn = False
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )

    @fx_wrap()
    def _process_input(self, inputs):
        if isinstance(inputs, placeholder):
            inputs = inputs.sample
        return inputs

    @fx_wrap()
    def _cache(self, feat, prev_feat):
        self.cache_history.data = prev_feat.detach()
        return feat

    def _fusion(self, feats, meta, compile_model):
        if compile_model is True or self.use_cache_history is True:
            if compile_model is True:
                prev_feat = self.quant(self._process_input(meta["prev_feat"]))
                prev_point = self._process_input(meta["prev_points"])
            else:
                n, c, h, w = feats.shape
                feats = feats.view(-1, self.num_seq, c, h, w)
                feats = feats[:, 0]
                prev_feat = self.quant(self.cache_history)
                prev_point = self._get_reference_points(prev_feat, meta, 0)

            prev_feat = self._transform(prev_feat, prev_point)
            fused_feat = self._fuse_op(prev_feat, feats)
            fused_feat = self.project(fused_feat)
        else:
            n, c, h, w = feats.shape
            feats = feats.view(-1, self.num_seq, c, h, w)
            prev = feats[:, -1]
            for i in reversed(range(0, self.num_seq - 1)):
                cur = feats[:, i]
                new_coords = self._get_reference_points(prev, meta, i)
                prev = self._transform(prev, new_coords)
                prev = self._fuse_op(prev, cur)
                prev = self.project(prev)
            fused_feat = prev
        return fused_feat


@OBJECT_REGISTRY.register
class AddTemporalFusion(TemporalFusion):
    """Simple Add Temporal fusion for bev feats."""

    def __init__(self, **kwargs):
        super(AddTemporalFusion, self).__init__(**kwargs)
        self.floatFs = FloatFunctional()

    def _fuse_op(self, prev, cur):
        return self.floatFs.add(prev, cur)
