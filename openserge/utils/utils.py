import torch, torch.nn.functional as F


def build_grid(h: int, w: int, device=None):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return x, y  # each [H, W]

def to_image_coords(u_off, v_off, cell_xy, stride, in_hw, out_hw):
    # u_off, v_off in [-0.5, 0.5], cell centers = (cell_xy + 0.5) * stride
    # Return x, y in input image coords
    cx = (cell_xy[..., 0] + 0.5 + u_off) * stride
    cy = (cell_xy[..., 1] + 0.5 + v_off) * stride
    # clamp to image size
    H, W = in_hw
    cx = cx.clamp(0, W-1)
    cy = cy.clamp(0, H-1)
    return cx, cy
