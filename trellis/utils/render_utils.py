import os 
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


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_snapshot_yaw_pitch(samples, resolution=512, bg_color=(255,255,255), yaw=[-16/180*np.pi], pitch=[20/180*np.pi], r=28, fov=2, **kwargs):
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)

def save_outputs(outputs, basedir, formats, tag=""):
    outdir = os.path.join(basedir, tag)
    os.makedirs(outdir, exist_ok=True)
    # Render the outputs
    
    if 'gaussian' in formats:
        total_view = render_snapshot_yaw_pitch(outputs['gaussian'][0], resolution=1024, yaw=[270/180*np.pi, 90/180*np.pi, 180/180*np.pi], pitch=[90/180*np.pi, 0*np.pi, 0*np.pi], r=28, fov=2)['color']
        # for i in range(len(top_view)):
        for i in range(len(total_view)):
            imageio.imwrite(os.path.join(outdir, f"sample_gs_view{i}.png"), total_view[i])
    
        # whole scene
        video = render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(outdir, f"sample_gs.mp4"), video, fps=30)
        # outputs['gaussian'][0].save_ply(outdir+f"sample.ply") # Save Gaussians as PLY files
        # each cube
        for i in range(1,len(outputs['gaussian'])):
            video = render_video(outputs['gaussian'][i])['color']
            imageio.mimsave(os.path.join(outdir, f"sample_gs{i}.mp4"), video, fps=30)
            # outputs['gaussian'][i].save_ply(outdir+f"sample{i}.ply") # Save Gaussians as PLY files

    if 'radiance_field' in formats:
        video = render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(os.path.join(outdir, "sample_rf.mp4"), video, fps=30)

    if 'mesh' in formats:
        video = render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(outdir, "sample_mesh.mp4"), video, fps=30)
        
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


