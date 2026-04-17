import os 
import hashlib
import imageio
import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence
from .evaluation_utils import extract_dinov2_image_features, extract_clip_image_features


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def _build_hammersley_eval_views(num_views=24, radius=2, fov=40, offset=(0.0, 0.0)):
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset=offset, remap=True)
        if p > 0:
            yaws.append(y)
            pitchs.append(p)
    rs = [radius] * len(yaws)
    fovs = [fov] * len(yaws)
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs)


def render_gs_eval_views(sample, resolution=512, bg_color=(0, 0, 0), num_views=24, radius=2, fov=40, offset=(0.0, 0.0), **kwargs):
    extrinsics, intrinsics = _build_hammersley_eval_views(num_views=num_views, radius=radius, fov=fov, offset=offset)
    rets = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
    return rets


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_snapshot_yaw_pitch(samples, resolution=512, bg_color=(0,0,0), yaw=[-16/180*np.pi], pitch=[20/180*np.pi], r=2, fov=40, **kwargs):
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def save_gaussian_through_iter(outputs, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Render the outputs
    videolist = []
    num_frames = 300
    for i, out in enumerate(outputs):
        video = np.concatenate([render_video(out['gaussian'][0], num_frames=num_frames)['color']], axis=0)
        videolist.append(video)
        imageio.mimsave(os.path.join(outdir, f"video_iter{i}.mp4"), video, fps=30)
        # out['gaussian'][0].save_ply(outdir+f"sample.ply") # Save Gaussians as PLY files
        
    idxlist = np.array_split(np.arange(num_frames), len(outputs))
    full_video = np.concatenate([videolist[i][idxlist[i]] for i in range(len(outputs))], axis=0)
    # import pdb; pdb.set_trace()
    imageio.mimsave(os.path.join(outdir, f"optimizevideo_throughiter{len(outputs)}.mp4"), full_video, fps=30)

def save_comparison_view(outputs, outdir, view_idx=None):
    os.makedirs(outdir, exist_ok=True)
    view_specs = [
        (270 / 180 * np.pi, 90 / 180 * np.pi),  # view0
        (90 / 180 * np.pi, 0.0),                # view1
        (0.0, 0.0),                              # view2
        (45 / 180 * np.pi, 30 / 180 * np.pi),   # view3
    ]
    if view_idx is None:
        yaw, pitch = (0 / 180 * np.pi, 30 / 180 * np.pi)
    else:
        if view_idx < 0 or view_idx >= len(view_specs):
            raise ValueError(f"view_idx must be in [0, {len(view_specs) - 1}], got {view_idx}")
        yaw, pitch = view_specs[view_idx]
    for i, out in enumerate(outputs):
        if out is None or ('gaussian' in out and (len(out['gaussian']) == 0 or out['gaussian'][0] is None)):
            total_view = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            total_view = render_snapshot_yaw_pitch(
                out['gaussian'][0],
                resolution=512,
                bg_color=(0,0,0),
                yaw=[yaw],
                pitch=[pitch],
                r=2,
                fov=40,
                verbose=False
            )['color'][0]
        imageio.imwrite(os.path.join(outdir, f"iter{i}.png" if len(outputs)>1 else f"base.png"), total_view)

def _to_preserve_mask(mask, device):
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if not torch.is_tensor(mask):
        raise ValueError("preserve_mask must be a torch.Tensor or numpy.ndarray.")
    if mask.ndim == 5:
        mask = mask[0, 0]
    elif mask.ndim == 4:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError("preserve_mask must have shape (H,W,D) or (1,1,H,W,D).")
    return (mask.to(device=device).float() < 0.5)


def _to_edit_mask(mask, device):
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if not torch.is_tensor(mask):
        raise ValueError("edit_mask must be a torch.Tensor or numpy.ndarray.")
    if mask.ndim == 5:
        mask = mask[0, 0]
    elif mask.ndim == 4:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError("edit_mask must have shape (H,W,D) or (1,1,H,W,D).")
    return (mask.to(device=device).float() > 0.5)


def _extract_preserve_region_mesh(mesh: MeshExtractResult, preserve_mask: torch.Tensor) -> MeshExtractResult:
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return MeshExtractResult(mesh.vertices, mesh.faces, mesh.vertex_attrs, mesh.res)

    reso = preserve_mask.shape[0]
    verts = mesh.vertices
    faces = mesh.faces

    # Mesh vertices are in roughly [-0.5, 0.5].
    vidx = torch.floor((verts + 0.5) * reso).long().clamp(0, reso - 1)
    keep_v = preserve_mask[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    keep_f = keep_v[faces].all(dim=1)
    kept_faces = faces[keep_f]

    if kept_faces.shape[0] == 0:
        empty_v = verts[:0]
        empty_f = faces[:0]
        empty_attr = mesh.vertex_attrs[:0] if mesh.vertex_attrs is not None else None
        return MeshExtractResult(empty_v, empty_f, empty_attr, mesh.res)

    unique_vidx, new_faces_flat = torch.unique(kept_faces.reshape(-1), sorted=True, return_inverse=True)
    new_faces = new_faces_flat.reshape(-1, 3)
    new_verts = verts[unique_vidx]
    new_attrs = mesh.vertex_attrs[unique_vidx] if mesh.vertex_attrs is not None else None
    return MeshExtractResult(new_verts, new_faces, new_attrs, mesh.res)


def _extract_preserve_region_gaussian(gs: Gaussian, preserve_mask: torch.Tensor) -> Gaussian:
    """
    Keep preserve-region gaussians by suppressing out-of-mask opacities.
    """
    xyz = gs.get_xyz
    reso = preserve_mask.shape[0]
    pmin, pmax = float(xyz.min().item()), float(xyz.max().item())
    if pmin < -0.1 or pmax > 1.1:
        xyz_unit = xyz + 0.5
    else:
        xyz_unit = xyz
    idx = torch.floor(xyz_unit * reso).long().clamp(0, reso - 1)
    keep = preserve_mask[idx[:, 0], idx[:, 1], idx[:, 2]]

    gs_recon = Gaussian(**gs.init_params)
    gs_recon._xyz = gs._xyz.clone()
    gs_recon._features_dc = gs._features_dc.clone()
    gs_recon._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    gs_recon._scaling = gs._scaling.clone()
    gs_recon._rotation = gs._rotation.clone()
    gs_recon._opacity = gs._opacity.clone()
    gs_recon._opacity[~keep] = -20.0
    return gs_recon


def _extract_edit_region_gaussian(gs: Gaussian, edit_mask: torch.Tensor) -> Gaussian:
    """
    Keep edit-region gaussians by suppressing out-of-mask opacities.
    """
    xyz = gs.get_xyz
    reso = edit_mask.shape[0]
    pmin, pmax = float(xyz.min().item()), float(xyz.max().item())
    if pmin < -0.1 or pmax > 1.1:
        xyz_unit = xyz + 0.5
    else:
        xyz_unit = xyz
    idx = torch.floor(xyz_unit * reso).long().clamp(0, reso - 1)
    keep = edit_mask[idx[:, 0], idx[:, 1], idx[:, 2]]

    gs_gen = Gaussian(**gs.init_params)
    gs_gen._xyz = gs._xyz.clone()
    gs_gen._features_dc = gs._features_dc.clone()
    gs_gen._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    gs_gen._scaling = gs._scaling.clone()
    gs_gen._rotation = gs._rotation.clone()
    gs_gen._opacity = gs._opacity.clone()
    gs_gen._opacity[~keep] = -20.0
    return gs_gen


def _stable_random_view_index(outdir: str, num_views: int, suffix: str = "") -> int:
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    key = f"{os.path.abspath(outdir)}::{suffix}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % num_views


def save_outputs(outputs, basedir, formats, tag="", preserve_mask=None):
    outdir = os.path.join(basedir, tag)
    os.makedirs(outdir, exist_ok=True)
    # Render the outputs
    
    if 'gaussian' in formats:
        # snapshot rendering
        total_view = render_snapshot_yaw_pitch(outputs['gaussian'][0], resolution=512, bg_color=(0,0,0), yaw=[270/180*np.pi, 90/180*np.pi, 0/180*np.pi, 45/180*np.pi], pitch=[90/180*np.pi, 0*np.pi, 0*np.pi, 30/180*np.pi], r=2, fov=40)['color']
        for i in range(len(total_view)):
            imageio.imwrite(os.path.join(outdir, f"sample_gs_view{i}.png"), total_view[i])
    
        # whole scene video for visualization
        video = render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(outdir, f"sample_gs.mp4"), video, fps=30)

        # Evaluation views (24 Hammersley cameras -> 18 upper hemisphere views).
        eval_dir = os.path.join(outdir, "eval_views")
        full_eval_dir = os.path.join(eval_dir, "full")
        os.makedirs(full_eval_dir, exist_ok=True)
        full_eval = render_gs_eval_views(outputs['gaussian'][0], num_views=24)['color']
        video_np = np.stack(full_eval, axis=0)
        for i, frame in enumerate(full_eval):
            imageio.imwrite(os.path.join(full_eval_dir, f"{i:03d}.png"), frame)

        # Save per-view DINOv2 / CLIP features for full-scene GS eval renders.
        dinov2_dir = os.path.join(outdir, "dinov2")
        clip_dir = os.path.join(outdir, "clip")
        os.makedirs(dinov2_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)
        full_dino = extract_dinov2_image_features(video_np)
        np.savez_compressed(
            os.path.join(dinov2_dir, "full.npz"),
            features=full_dino,
            pooled=full_dino.mean(axis=0).astype(np.float32),
        )
        full_random_idx = _stable_random_view_index(outdir, full_dino.shape[0], suffix="full")
        np.savez_compressed(
            os.path.join(dinov2_dir, "random_view_full.npz"),
            feature=full_dino[full_random_idx].astype(np.float32),
            view_index=np.int64(full_random_idx),
            num_views=np.int64(full_dino.shape[0]),
        )
        full_clip = extract_clip_image_features(video_np)
        np.savez_compressed(
            os.path.join(clip_dir, "full.npz"),
            features=full_clip,
            pooled=full_clip.mean(axis=0).astype(np.float32),
        )

        if preserve_mask is not None:
            preserve_mask_3d = _to_preserve_mask(preserve_mask, outputs['gaussian'][0].get_xyz.device)
            recon_gs = _extract_preserve_region_gaussian(outputs['gaussian'][0], preserve_mask_3d)
            recon_video = render_video(recon_gs)['color']
            imageio.mimsave(os.path.join(outdir, "sample_gs_recon.mp4"), recon_video, fps=30)
            edit_mask_3d = _to_edit_mask(preserve_mask, outputs['gaussian'][0].get_xyz.device)
            gen_gs = _extract_edit_region_gaussian(outputs['gaussian'][0], edit_mask_3d)
            gen_video = render_video(gen_gs)['color']
            imageio.mimsave(os.path.join(outdir, "sample_gs_gen.mp4"), gen_video, fps=30)

            recon_eval_dir = os.path.join(eval_dir, "recon")
            gen_eval_dir = os.path.join(eval_dir, "gen")
            os.makedirs(recon_eval_dir, exist_ok=True)
            os.makedirs(gen_eval_dir, exist_ok=True)
            recon_eval = render_gs_eval_views(recon_gs, num_views=24)['color']
            gen_eval = render_gs_eval_views(gen_gs, num_views=24)['color']
            for i, frame in enumerate(recon_eval):
                imageio.imwrite(os.path.join(recon_eval_dir, f"{i:03d}.png"), frame)
            for i, frame in enumerate(gen_eval):
                imageio.imwrite(os.path.join(gen_eval_dir, f"{i:03d}.png"), frame)

            # Save edit-region-only features from masked GS eval renders.
            gen_np = np.stack(gen_eval, axis=0)
            gen_dino = extract_dinov2_image_features(gen_np)
            np.savez_compressed(
                os.path.join(dinov2_dir, "edit_only.npz"),
                features=gen_dino,
                pooled=gen_dino.mean(axis=0).astype(np.float32),
            )
            edit_random_idx = _stable_random_view_index(outdir, gen_dino.shape[0], suffix="edit_only")
            np.savez_compressed(
                os.path.join(dinov2_dir, "random_view_edit_only.npz"),
                feature=gen_dino[edit_random_idx].astype(np.float32),
                view_index=np.int64(edit_random_idx),
                num_views=np.int64(gen_dino.shape[0]),
            )
            gen_clip = extract_clip_image_features(gen_np)
            np.savez_compressed(
                os.path.join(clip_dir, "edit_only.npz"),
                features=gen_clip,
                pooled=gen_clip.mean(axis=0).astype(np.float32),
            )
        
        # Save Gaussians as PLY files
        outputs['gaussian'][0].save_ply(outdir+f"sample.ply") 

    if 'radiance_field' in formats:
        video = render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(os.path.join(outdir, "sample_rf.mp4"), video, fps=30)

    if 'mesh' in formats:
        video = render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(outdir, "sample_mesh.mp4"), video, fps=30)
        if preserve_mask is not None:
            preserve_mask_3d = _to_preserve_mask(preserve_mask, outputs['mesh'][0].vertices.device)
            recon_mesh = _extract_preserve_region_mesh(outputs['mesh'][0], preserve_mask_3d)
            video_recon = render_video(recon_mesh)['normal']
            imageio.mimsave(os.path.join(outdir, "sample_mesh_recon.mp4"), video_recon, fps=30)
        
    if 'gaussian' in formats and 'mesh' in formats:
        from . import postprocessing_utils
        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(outdir, "sample.glb"))
        
    return 0
