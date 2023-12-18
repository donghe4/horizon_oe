import logging
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.nn.functional import grid_sample

from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils import _log_first_n, fx_helper


######################################################################
# torch grid_sample raises Exception with cuda bfloat16 amp
# TODO: remove this after grid_sample support cuda bfloat16
@fx_helper.wrap()
def autocasted_grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    if input.device.type == "cuda" and input.dtype == torch.bfloat16:
        input = input.float()
    return grid_sample(
        input,
        grid,
        mode,
        padding_mode,
        align_corners,
    )


# use a outer func to help fx correctly find the wrapped inner
def autocasted_grid_sample_outer(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    return autocasted_grid_sample(
        input,
        grid,
        mode,
        padding_mode,
        align_corners,
    )


QTensor.patch_torch_func(
    torch.nn.functional.grid_sample, autocasted_grid_sample_outer
)

torch.nn.functional.grid_sample = autocasted_grid_sample_outer
######################################################################


class GridSample(torch.nn.Module):
    """Refine this docstring in the future.

    Given an input and a flow-field grid, computes the output using
    input values and pixel locations from grid.

    Note that the grid required by this function is DIFFERENT from
    torch.nn.functional.grid_sample !!!

    Args:
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" and "nearest" supported now.
            Defaults to "bilinear".
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" and "border" is supported now.
            Defaults to "border".
        align_corners ([type], optional): Since the grid format is
            different with torch.nn.functional.grid_sample, this param
            does not have any effect now.
            Defaults to None.
    """

    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=False
    ):
        super(GridSample, self).__init__()

        assert mode in (
            "bilinear",
            "nearest",
        ), "GridSample only support 'bilinear' and 'nearest' mode now"
        assert padding_mode in (
            "zeros",
            "border",
        ), "GridSample only support 'zeros' and 'border' padding_mode now"
        assert isinstance(
            align_corners, (bool, type(None))
        ), "param 'align_corners' must be bool or None"

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        _log_first_n(
            logging.WARN,
            "GridSample module is deprecated,"
            "please use torch.nn.functional.grid_sample",
            name=__name__,
        )

    def forward(self, x, grid):
        # type: (Tensor, Tensor) -> Tensor
        """
        Forward pass of GridSample.

        Args:
            x (Tensor[N, C, H, W]): Input data.
            grid (Tensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
                is different with torch.nn.functional.grid_sample. In this
                function, the sample point of output point (x, y) is computed
                by (x + dx, y + dy).
        """
        # convert grid format from 'delta' to 'norm'
        n = grid.size(0)
        h = grid.size(1)
        w = grid.size(2)
        base_coord_y = (
            torch.arange(h, dtype=grid.dtype, device=grid.device)
            .unsqueeze(-1)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        base_coord_x = (
            torch.arange(w, dtype=grid.dtype, device=grid.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        absolute_grid_x = grid[:, :, :, 0] + base_coord_x
        absolute_grid_y = grid[:, :, :, 1] + base_coord_y
        norm_grid_x = absolute_grid_x * 2 / (x.size(3) - 1) - 1
        norm_grid_y = absolute_grid_y * 2 / (x.size(2) - 1) - 1
        norm_grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1)

        r = F.grid_sample(x, norm_grid, self.mode, self.padding_mode, True)
        return r
