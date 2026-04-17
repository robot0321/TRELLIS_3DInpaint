import torch
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from typing import Union, Dict
import imageio
from PIL import Image

def get_keepidx(xyz, mask):
    assert mask.ndim==3 and len(set(mask.shape[-3:])) # same H,W,D reso
    reso=mask.shape[-1]
    out_idx=torch.floor(xyz*reso).long().clamp(0,reso-1)
    ix,iy,iz=out_idx.unbind(dim=-1)
    return (mask[ix,iy,iz]<0.5)

def evaluate_optinit(base_slat, optinit_slat, optinit_mask):
    geometry_metrics = compute_geometry_metrics(base_slat, optinit_slat, optinit_mask)
    appearance_metrics = compute_appearance_metrics(base_slat, optinit_slat, optinit_mask)
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

def compute_geometry_metrics(base_slat, optinit_slat, optinit_mask):
    """
    Compute geometry metrics between base and optinit SLATs within the optinit region.

    Args:
        base_slat (torch.Tensor): Base SLAT tensor of shape (C, H, W, D).
        optinit_slat (torch.Tensor): Optinit SLAT tensor of shape (C, H, W, D).
        optinit_mask (torch.Tensor): Optinit mask of shape (1, 1, H, W, D) with 1s in the region to evaluate.

    Returns:
        dict: Dictionary containing geometry metrics.
    """
    # Extract coordinates of the optinit region
    mask_coords = torch.nonzero(optinit_mask[0, 0], as_tuple=False)  # (N, 3)

    # Extract point clouds from SLATs within the masked region
    base_points = base_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)
    optinit_points = optinit_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)

    # Compute Chamfer Distance
    chamfer_dist = compute_chamfer_distance(base_points[:, :3], optinit_points[:, :3])

    return {
        "chamfer_distance": chamfer_dist,
    }
    
def compute_appearance_metrics(base_slat, optinit_slat, optinit_mask):
    """
    Compute appearance metrics between base and optinit SLATs within the optinit region.

    Args:
        base_slat (torch.Tensor): Base SLAT tensor of shape (C, H, W, D).
        optinit_slat (torch.Tensor): Optinit SLAT tensor of shape (C, H, W, D).
        optinit_mask (torch.Tensor): Optinit mask of shape (1, 1, H, W, D) with 1s in the region to evaluate.

    Returns:
        dict: Dictionary containing appearance metrics.
    """
    # Extract coordinates of the optinit region
    mask_coords = torch.nonzero(optinit_mask[0, 0], as_tuple=False)  # (N, 3)

    # Extract colors from SLATs within the masked region
    base_colors = base_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)
    optinit_colors = optinit_slat[:, mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]].T  # (N, C)

    # Compute Mean Squared Error (MSE) for colors
    mse = torch.mean((base_colors[:, 3:] - optinit_colors[:, 3:]) ** 2).item()

    return {
        "mse_color": mse,
    }

def compute_chamfer_distance(base_z, optinit_z, optinit_mask):
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


def _load_glb_mesh(path: str) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(path, force='scene')
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = []
        for geom in mesh_or_scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom)
        if len(meshes) == 0:
            raise ValueError(f"No valid mesh geometry found in GLB: {path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise ValueError(f"Unsupported GLB type at {path}: {type(mesh_or_scene)}")


def _sample_mesh_points(mesh: trimesh.Trimesh, num_points: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def _mask_to_3d(mask: Union[str, torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(mask, str):
        mask = torch.load(mask)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 5:
        mask = mask[0, 0]
    elif mask.ndim == 4:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError("mask must be shape (H,W,D) or (1,1,H,W,D).")
    return mask.astype(np.float32)


def _points_keep_mask(points: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """
    Keep points in preserve region (mask < 0.5).
    Supports two common coordinate conventions:
    - unit cube [0, 1]
    - centered cube [-0.5, 0.5]
    """
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    reso = mask_3d.shape[0]
    pmin, pmax = float(points.min()), float(points.max())

    # Auto-detect coordinate normalization.
    if pmin < -0.1 or pmax > 1.1:
        # Assume centered cube coordinates in [-0.5, 0.5].
        points_unit = points + 0.5
    else:
        # Assume unit cube coordinates in [0, 1].
        points_unit = points

    idx = np.floor(points_unit * reso).astype(np.int64)
    idx = np.clip(idx, 0, reso - 1)
    return mask_3d[idx[:, 0], idx[:, 1], idx[:, 2]] < 0.5


def _chamfer_fscore(points_a: np.ndarray, points_b: np.ndarray, tau_list) -> Dict[str, float]:
    if isinstance(tau_list, (float, int)):
        tau_list = [float(tau_list)]
    tau_list = [float(t) for t in tau_list]

    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        out = {
            "chamfer_l2": float("inf"),
            "chamfer_l1": float("inf"),
        }
        for tau in tau_list:
            out[f"fscore@{tau}"] = 0.0
            out[f"precision@{tau}"] = 0.0
            out[f"recall@{tau}"] = 0.0
        return out

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_a, _ = tree_b.query(points_a, k=1, workers=-1)  # A -> B
    d_b, _ = tree_a.query(points_b, k=1, workers=-1)  # B -> A

    chamfer_l2 = float(np.mean(d_a ** 2) + np.mean(d_b ** 2))
    chamfer_l1 = float(np.mean(d_a) + np.mean(d_b))
    out = {
        "chamfer_l2": chamfer_l2,
        "chamfer_l1": chamfer_l1,
    }
    for tau in tau_list:
        precision = float(np.mean(d_b < tau))
        recall = float(np.mean(d_a < tau))
        fscore = float(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
        out[f"fscore@{tau}"] = fscore
        out[f"precision@{tau}"] = precision
        out[f"recall@{tau}"] = recall
    return out


def evaluate_optinit_glb_preserve_region(
    base_glb_path: str,
    edited_glb_path: str,
    mask: Union[str, torch.Tensor, np.ndarray],
    num_points: int = 200000,
    tau_list: Union[list, tuple, float] = (0.005, 0.01, 0.02),
    seed: int = 0,
) -> Dict[str, float]:
    """
    Evaluate geometry consistency in preserve region (mask==0) between two GLBs.

    Args:
        base_glb_path: Path to base sample.glb.
        edited_glb_path: Path to edited/repaint sample.glb.
        mask: 3D binary mask tensor/path. 1 means edited region, 0 means preserve region.
        num_points: Number of surface samples per mesh.
        tau_list: Distance thresholds for F-score.
        seed: Random seed for surface sampling.
    """
    mask_3d = _mask_to_3d(mask)
    base_mesh = _load_glb_mesh(base_glb_path)
    edited_mesh = _load_glb_mesh(edited_glb_path)

    base_pts = _sample_mesh_points(base_mesh, num_points=num_points, seed=seed)
    edited_pts = _sample_mesh_points(edited_mesh, num_points=num_points, seed=seed + 1)

    base_keep = _points_keep_mask(base_pts, mask_3d)
    edited_keep = _points_keep_mask(edited_pts, mask_3d)
    base_keep_pts = base_pts[base_keep]
    edited_keep_pts = edited_pts[edited_keep]

    metrics = _chamfer_fscore(base_keep_pts, edited_keep_pts, tau_list=tau_list)
    metrics.update({
        "base_points_total": int(base_pts.shape[0]),
        "edited_points_total": int(edited_pts.shape[0]),
        "base_points_preserve": int(base_keep_pts.shape[0]),
        "edited_points_preserve": int(edited_keep_pts.shape[0]),
    })
    return metrics


def evaluate_normal_video_psnr_lpips(
    base_video_path: str,
    edited_video_path: str,
    device: str = None,
) -> Dict[str, float]:
    """
    Compare two normal-render videos by PSNR and LPIPS.
    """
    base_video = imageio.v3.imread(base_video_path)
    edited_video = imageio.v3.imread(edited_video_path)

    if base_video.ndim != 4 or edited_video.ndim != 4:
        raise ValueError("Videos must be rank-4 arrays: (T, H, W, C).")
    if base_video.shape[-1] != 3 or edited_video.shape[-1] != 3:
        raise ValueError("Videos must have 3 channels.")

    nframes = min(base_video.shape[0], edited_video.shape[0])
    if nframes == 0:
        raise ValueError("One of the videos has zero frames.")
    base_video = base_video[:nframes]
    edited_video = edited_video[:nframes]

    if base_video.shape[1:] != edited_video.shape[1:]:
        raise ValueError(f"Video spatial size mismatch: {base_video.shape[1:]} vs {edited_video.shape[1:]}")

    base_f = base_video.astype(np.float32) / 255.0
    edited_f = edited_video.astype(np.float32) / 255.0

    # Frame-wise PSNR.
    mse = np.mean((base_f - edited_f) ** 2, axis=(1, 2, 3))
    psnr = np.where(mse > 1e-12, 10.0 * np.log10(1.0 / mse), 100.0)

    # Frame-wise LPIPS.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        lpips_vals = []
        with torch.no_grad():
            for i in range(nframes):
                b = torch.from_numpy(base_f[i]).permute(2, 0, 1).unsqueeze(0).to(device)
                e = torch.from_numpy(edited_f[i]).permute(2, 0, 1).unsqueeze(0).to(device)
                b = b * 2 - 1
                e = e * 2 - 1
                lpips_vals.append(float(lpips_model(b, e).item()))
        lpips_mean = float(np.mean(lpips_vals))
        lpips_std = float(np.std(lpips_vals))
    except Exception:
        lpips_mean = float("nan")
        lpips_std = float("nan")

    return {
        "num_frames": int(nframes),
        "psnr_mean": float(np.mean(psnr)),
        "psnr_std": float(np.std(psnr)),
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
    }


def evaluate_video_psnr_ssim_lpips(
    base_video_path: str,
    edited_video_path: str,
    device: str = None,
    prefix: str = "",
    bg_mode: str = "auto",
) -> Dict[str, float]:
    """
    Compare two RGB videos with PSNR, SSIM, and LPIPS.
    """
    base_video = imageio.v3.imread(base_video_path)
    edited_video = imageio.v3.imread(edited_video_path)

    if base_video.ndim != 4 or edited_video.ndim != 4:
        raise ValueError("Videos must be rank-4 arrays: (T, H, W, C).")
    if base_video.shape[-1] != 3 or edited_video.shape[-1] != 3:
        raise ValueError("Videos must have 3 channels.")

    nframes = min(base_video.shape[0], edited_video.shape[0])
    if nframes == 0:
        raise ValueError("One of the videos has zero frames.")
    base_video = base_video[:nframes]
    edited_video = edited_video[:nframes]
    if base_video.shape[1:] != edited_video.shape[1:]:
        raise ValueError(f"Video spatial size mismatch: {base_video.shape[1:]} vs {edited_video.shape[1:]}")

    base_f = base_video.astype(np.float32) / 255.0
    edited_f = edited_video.astype(np.float32) / 255.0

    if bg_mode == "auto":
        if prefix.startswith("geo-"):
            bg_mode = "geo"
        elif prefix.startswith("app-"):
            bg_mode = "app"
        else:
            bg_mode = "app"

    def _foreground_mask(frame: np.ndarray, mode: str) -> np.ndarray:
        """
        frame: (H, W, 3), float [0,1]
        returns bool mask where True means foreground.
        """
        black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        d_black = np.linalg.norm(frame - black[None, None, :], axis=-1)
        if mode == "app":
            bg = d_black < 0.06
        elif mode == "geo":
            d_gray = np.linalg.norm(frame - gray[None, None, :], axis=-1)
            bg = (d_black < 0.06) | (d_gray < 0.045)
        else:
            bg = d_black < 0.06
        return ~bg

    frame_pairs = []
    for i in range(nframes):
        fg_b = _foreground_mask(base_f[i], bg_mode)
        fg_e = _foreground_mask(edited_f[i], bg_mode)
        fg = fg_b | fg_e
        if not np.any(fg):
            continue
        ys, xs = np.where(fg)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        b_crop = base_f[i, y0:y1, x0:x1]
        e_crop = edited_f[i, y0:y1, x0:x1]
        fg_crop = fg[y0:y1, x0:x1]
        frame_pairs.append((b_crop, e_crop, fg_crop))

    if len(frame_pairs) == 0:
        def k(name: str) -> str:
            return f"{prefix}{name}" if prefix else name
        return {
            k("num_frames"): 0,
            k("psnr_mean"): float("nan"),
            k("psnr_std"): float("nan"),
            k("ssim_mean"): float("nan"),
            k("ssim_std"): float("nan"),
            k("lpips_mean"): float("nan"),
            k("lpips_std"): float("nan"),
        }

    # PSNR
    psnr_vals = []
    for b_crop, e_crop, fg_crop in frame_pairs:
        diff2 = (b_crop - e_crop) ** 2
        denom = np.maximum(float(fg_crop.sum()) * 3.0, 1.0)
        mse = float((diff2 * fg_crop[..., None]).sum() / denom)
        psnr_vals.append(100.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse)))
    psnr = np.array(psnr_vals, dtype=np.float32)

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_vals = []
        for b_crop, e_crop, _ in frame_pairs:
            ssim_vals.append(float(ssim_fn(b_crop, e_crop, data_range=1.0, channel_axis=2)))
        ssim_mean = float(np.mean(ssim_vals))
        ssim_std = float(np.std(ssim_vals))
    except Exception:
        ssim_mean = float("nan")
        ssim_std = float("nan")

    # LPIPS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        lpips_vals = []
        with torch.no_grad():
            for b_crop, e_crop, _ in frame_pairs:
                b = torch.from_numpy(b_crop).permute(2, 0, 1).unsqueeze(0).to(device)
                e = torch.from_numpy(e_crop).permute(2, 0, 1).unsqueeze(0).to(device)
                b = b * 2 - 1
                e = e * 2 - 1
                lpips_vals.append(float(lpips_model(b, e).item()))
        lpips_mean = float(np.mean(lpips_vals))
        lpips_std = float(np.std(lpips_vals))
    except Exception:
        lpips_mean = float("nan")
        lpips_std = float("nan")

    def k(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    return {
        k("num_frames"): int(len(frame_pairs)),
        k("psnr_mean"): float(np.mean(psnr)),
        k("psnr_std"): float(np.std(psnr)),
        k("ssim_mean"): ssim_mean,
        k("ssim_std"): ssim_std,
        k("lpips_mean"): lpips_mean,
        k("lpips_std"): lpips_std,
    }


def evaluate_clip_video_text(
    video_path: str,
    text: str,
    model_name: str = "openai/clip-vit-base-patch32",
    max_frames: int = 32,
    device: str = None,
    prefix: str = "gen-",
) -> Dict[str, float]:
    """
    Compute CLIP similarity between video frames and text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    video = imageio.v3.imread(video_path)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("Video must be (T,H,W,3).")
    if video.shape[0] == 0:
        raise ValueError("Video has zero frames.")

    # Uniformly subsample frames for speed.
    n = video.shape[0]
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).astype(np.int64)
        frames = video[idx]
    else:
        frames = video

    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "clean_up_tokenization_spaces"):
            processor.tokenizer.clean_up_tokenization_spaces = False

        inputs = processor(
            text=[text],
            images=[f for f in frames],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(-1).detach().cpu().numpy()

        return {
            f"{prefix}clipscore_mean": float(np.mean(sims)),
            f"{prefix}clipscore_std": float(np.std(sims)),
            f"{prefix}clipscore_num_frames": int(frames.shape[0]),
        }
    except Exception:
        return {
            f"{prefix}clipscore_mean": float("nan"),
            f"{prefix}clipscore_std": float("nan"),
            f"{prefix}clipscore_num_frames": int(frames.shape[0]),
        }


_DINO_MODEL = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def extract_dinov2_image_features(
    frames: np.ndarray,
    model_name: str = "dinov2_vitl14_reg",
    batch_size: int = 8,
    device: str = None,
) -> np.ndarray:
    """
    Extract per-frame DINOv2 features from RGB frames.
    Returns (N, D) float32.
    """
    global _DINO_MODEL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _DINO_MODEL is None:
        _DINO_MODEL = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be (N,H,W,3).")
    if frames.shape[0] == 0:
        return np.zeros((0, 1024), dtype=np.float32)

    imgs = []
    for f in frames:
        img = Image.fromarray(f.astype(np.uint8)).resize((518, 518), Image.Resampling.LANCZOS)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(2, 0, 1)
        imgs.append(arr)

    x = torch.stack(imgs, dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean.cpu()) / std.cpu()

    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size].to(device)
            out = _DINO_MODEL(xb, is_training=True)
            if isinstance(out, dict):
                if "x_norm_clstoken" in out:
                    fb = out["x_norm_clstoken"]
                elif "x_prenorm" in out:
                    fb = out["x_prenorm"][:, 0]
                else:
                    raise ValueError("Unexpected DINOv2 output dict keys.")
            else:
                fb = out
            feats.append(fb.detach().cpu().float().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_clip_image_features(
    frames: np.ndarray,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    device: str = None,
) -> np.ndarray:
    """
    Extract per-frame CLIP image features from RGB frames.
    Returns L2-normalized (N, D) float32.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        from transformers import CLIPProcessor, CLIPModel
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        if hasattr(_CLIP_PROCESSOR, "tokenizer") and hasattr(_CLIP_PROCESSOR.tokenizer, "clean_up_tokenization_spaces"):
            _CLIP_PROCESSOR.tokenizer.clean_up_tokenization_spaces = False

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be (N,H,W,3).")
    if frames.shape[0] == 0:
        return np.zeros((0, 512), dtype=np.float32)

    feats = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch_size):
            fb = frames[i:i + batch_size]
            inputs = _CLIP_PROCESSOR(images=[x for x in fb], return_tensors="pt", padding=True).to(device)
            image_features = _CLIP_MODEL.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feats.append(image_features.detach().cpu().float().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def _frechet_distance_diag(mu_a: np.ndarray, var_a: np.ndarray, mu_b: np.ndarray, var_b: np.ndarray) -> float:
    """
    Diagonal-covariance Frechet distance.
    """
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    var_a = np.asarray(var_a, dtype=np.float64)
    var_b = np.asarray(var_b, dtype=np.float64)
    term_mean = np.sum((mu_a - mu_b) ** 2)
    term_var = np.sum(var_a + var_b - 2.0 * np.sqrt(np.maximum(var_a * var_b, 0.0)))
    return float(term_mean + term_var)


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def _mmd2_rbf(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        return float("nan")

    dxy = _pairwise_sq_dists(x, y)
    if sigma is None:
        all_xy = np.concatenate([x, y], axis=0)
        d_all = _pairwise_sq_dists(all_xy, all_xy)
        tri = d_all[np.triu_indices_from(d_all, k=1)]
        med = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        sigma = float(np.sqrt(max(med, 1e-12)))
    gamma = 1.0 / (2.0 * sigma * sigma)

    kxy = np.exp(-gamma * dxy)
    dxx = _pairwise_sq_dists(x, x)
    dyy = _pairwise_sq_dists(y, y)
    kxx = np.exp(-gamma * dxx)
    kyy = np.exp(-gamma * dyy)

    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    mmd2 = (
        kxx.sum() / (nx * (nx - 1))
        + kyy.sum() / (ny * (ny - 1))
        - 2.0 * kxy.mean()
    )
    return float(mmd2)


def evaluate_fd_kd_from_dinov2_features(
    generated_features: np.ndarray,
    dataset_features: np.ndarray,
    dataset_mean: np.ndarray,
    dataset_variance: np.ndarray,
    max_samples: int = 100,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Evaluate FD (diagonal Frechet) and KD (MMD^2 with RBF kernel) on DINOv2 features.
    KD uses equal-sized subsamples from dataset/generated, capped by max_samples.
    """
    gen = np.asarray(generated_features, dtype=np.float32)
    data = np.asarray(dataset_features, dtype=np.float32)
    if gen.ndim != 2 or data.ndim != 2:
        raise ValueError("generated_features and dataset_features must be rank-2.")
    if gen.shape[0] == 0 or data.shape[0] == 0:
        return {"fd": float("nan"), "kd": float("nan"), "num_gen": int(gen.shape[0]), "num_kd_samples": 0}

    gen_mean = gen.mean(axis=0, dtype=np.float64)
    gen_var = gen.var(axis=0, dtype=np.float64)
    fd = _frechet_distance_diag(gen_mean, gen_var, dataset_mean, dataset_variance)

    rng = np.random.default_rng(seed)
    n = min(int(max_samples), gen.shape[0], data.shape[0])
    gen_idx = rng.choice(gen.shape[0], size=n, replace=False) if gen.shape[0] > n else np.arange(gen.shape[0])
    data_idx = rng.choice(data.shape[0], size=n, replace=False) if data.shape[0] > n else np.arange(data.shape[0])
    kd = _mmd2_rbf(gen[gen_idx], data[data_idx])

    return {
        "fd": float(fd),
        "kd": float(kd),
        "num_gen": int(gen.shape[0]),
        "num_kd_samples": int(n),
    }
