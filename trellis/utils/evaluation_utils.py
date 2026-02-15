import torch
import numpy as np
from scipy.spatial import cKDTree

def get_keepidx(xyz, mask):
    assert mask.ndim==3 and len(set(mask.shape[-3:])) # same H,W,D reso
    reso=mask.shape[-1]
    out_idx=torch.floor(xyz*reso).long().clamp(0,reso-1)
    ix,iy,iz=out_idx.unbind(dim=-1)
    return (mask[ix,iy,iz]<0.5)

def evaluate_inpainting(base_slat, inpaint_slat, inpaint_mask):
    geometry_metrics = compute_geometry_metrics(base_slat, inpaint_slat, inpaint_mask)
    appearance_metrics = compute_appearance_metrics(base_slat, inpaint_slat, inpaint_mask)
    return {**geometry_metrics, **appearance_metrics}


def compute_iou_and_dice(occA, occB):
    # iou/dice in grid
    # occA, occB: boolean tensor (B,1,H,W,D)
    assert occA.shape == occB.shape, "Occupancy tensors must have the same shape."
    assert occA.dtype == torch.bool and occB.dtype == torch.bool, "Occupancy tensors must be boolean."
    inter = torch.logical_and(occA, occB).sum()
    union = torch.logical_or(occA, occB).sum()
    iou = inter / (union + 1e-8)
    dice = 2 * inter / (occA.sum() + occB.sum() + 1e-8)
    return iou, dice

def compare_geo_color(
    xyzA, rgbA, xyzB, rgbB, 
    tau_list=[0.005, 0.01, 0.02]
):
    xyzA, rgbA = np.asarray(xyzA), np.asarray(rgbA)
    xyzB, rgbB = np.asarray(xyzB), np.asarray(rgbB)
    treeA = cKDTree(xyzA)
    treeB = cKDTree(xyzB)

    dA, idxA = treeB.query(xyzA, k=1, workers=-1)
    dB, idxB = treeA.query(xyzB, k=1, workers=-1)

    results = {}
    for tau in tau_list:
        P = np.mean(dB < tau)   # Precision
        R = np.mean(dA < tau)   # Recall
        F = 0 if (P+R)==0 else 2*P*R/(P+R)
        results[f"F@{tau}"] = F

    # Color error on geometry matches (A->B + B->A)
    maskA = (dA < tau_list[-1])
    maskB = (dB < tau_list[-1])

    col_err_A = np.linalg.norm(rgbA[maskA] - rgbB[idxA[maskA]], axis=1)
    col_err_B = np.linalg.norm(rgbB[maskB] - rgbA[idxB[maskB]], axis=1)

    results["color_mean"]   = np.mean(np.concatenate([col_err_A, col_err_B]))
    results["color_median"] = np.median(np.concatenate([col_err_A, col_err_B]))

    return results

def compute_geometry_metrics(base_slat, inpaint_slat, inpaint_mask):
    """
    Compute geometry metrics between base and inpainted SLATs within the inpainted region.

    Args:
        base_slat (torch.Tensor): Base SLAT tensor of shape (C, H, W, D).
        inpaint_slat (torch.Tensor): Inpainted SLAT tensor of shape (C, H, W, D).
        inpaint_mask (torch.Tensor): Inpainting mask of shape (1, 1, H, W, D) with 1s in the region to evaluate.

    Returns:
        dict: Dictionary containing geometry metrics.
    """
    # Extract coordinates of the inpainted region
    mask_coords = torch.nonzero(inpaint_mask[0, 0], as_tuple=False)  # (N, 3)

    # Extract point clouds from SLATs within the masked region
    base_points = base_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)
    inpaint_points = inpaint_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)

    # Compute Chamfer Distance
    chamfer_dist = compute_chamfer_distance(base_points[:, :3], inpaint_points[:, :3])

    return {
        "chamfer_distance": chamfer_dist,
    }
    
def compute_appearance_metrics(base_slat, inpaint_slat, inpaint_mask):
    """
    Compute appearance metrics between base and inpainted SLATs within the inpainted region.

    Args:
        base_slat (torch.Tensor): Base SLAT tensor of shape (C, H, W, D).
        inpaint_slat (torch.Tensor): Inpainted SLAT tensor of shape (C, H, W, D).
        inpaint_mask (torch.Tensor): Inpainting mask of shape (1, 1, H, W, D) with 1s in the region to evaluate.

    Returns:
        dict: Dictionary containing appearance metrics.
    """
    # Extract coordinates of the inpainted region
    mask_coords = torch.nonzero(inpaint_mask[0, 0], as_tuple=False)  # (N, 3)

    # Extract colors from SLATs within the masked region
    base_colors = base_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)
    inpaint_colors = inpaint_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)

    # Compute Mean Squared Error (MSE) for colors
    mse = torch.mean((base_colors[:, 3:] - inpaint_colors[:, 3:]) ** 2).item()

    return {
        "mse_color": mse,
    }

def compute_chamfer_distance(base_z, inpaint_z, inpaint_mask):
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        point_cloud_pred (torch.Tensor): Predicted point cloud of shape (N, 3).
        point_cloud_gt (torch.Tensor): Ground truth point cloud of shape (M, 3).

    Returns:
        float: Chamfer Distance value.
    """
    from pytorch3d.loss import chamfer_distance

    pc_pred = point_cloud_pred.unsqueeze(0)  # Add batch dimension
    pc_gt = point_cloud_gt.unsqueeze(0)      # Add batch dimension

    chamfer_dist, _ = chamfer_distance(pc_pred, pc_gt)
    return chamfer_dist.item()