import torch
def evaluate_inpainting(base_slat, inpaint_slat, inpaint_mask):
    geometry_metrics = compute_geometry_metrics(base_slat, inpaint_slat, inpaint_mask)
    appearance_metrics = compute_appearance_metrics(base_slat, inpaint_slat, inpaint_mask)
    return {**geometry_metrics, **appearance_metrics}

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