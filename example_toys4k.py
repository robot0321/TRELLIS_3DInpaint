import argparse
import ast
import csv
import os
import re
import time
from typing import Dict, List, Tuple

os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

METHODS = ("optinit", "blendeddiffusion", "md", "ilvr", "sdedit", "repaint", "dps")


def prompt_slug(text: str, max_len: int = 120) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', text.strip())
    s = s.strip('._-')
    if not s:
        s = 'empty_prompt'
    return s[:max_len]


def detach_tree(x, _visited=None):
    if _visited is None:
        _visited = set()

    if torch.is_tensor(x):
        return x.detach()

    oid = id(x)
    if oid in _visited:
        return x
    _visited.add(oid)

    if isinstance(x, dict):
        return {k: detach_tree(v, _visited) for k, v in x.items()}
    if isinstance(x, list):
        return [detach_tree(v, _visited) for v in x]
    if isinstance(x, tuple):
        return tuple(detach_tree(v, _visited) for v in x)

    # Handle custom objects like Gaussian / MeshExtractResult / Strivec containers.
    if hasattr(x, '__dict__'):
        for k, v in list(vars(x).items()):
            setattr(x, k, detach_tree(v, _visited))
        return x

    return x


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Toys4k latents with multiple inpainting baselines.')
    p.add_argument('--dataset_name', type=str, default=None,
                   help='Dataset folder name under datasets/, e.g. Toys4k_fullview, HSSD_fullview')
    p.add_argument('--metadata_csv', type=str, default='datasets/Toys4k_fullview/metadata.csv')
    p.add_argument('--ss_latent_root', type=str, default='datasets/Toys4k_fullview/ss_latents')
    p.add_argument('--latent_root', type=str, default='datasets/Toys4k_fullview/latents')
    p.add_argument('--ss_latent_model', type=str, default='ss_enc_conv3d_16l8_fp16')
    p.add_argument('--latent_model', type=str, default='dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16')
    p.add_argument('--mask', type=str, default='mask_preset/righthalf.pt')
    p.add_argument('--tag', type=str, default='eval_toys4k')

    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--optinit_out_subdir', type=str, default='optinit', help='Output subdirectory name for optinit results.')
    p.add_argument('--sha256', type=str, default=None, help='Run only a single sha256 sample.')
    p.add_argument('--sha256_file', type=str, default=None, help='Run only sha256s listed in this file (one per line).')
    p.add_argument('--limit', type=int, default=None, help='Max number of valid metadata rows to evaluate.')
    p.add_argument('--offset', type=int, default=0, help='Skip first N valid metadata rows.')

    p.add_argument('--seed', type=int, default=0, help='Edit seed for all methods.')
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--cfg_ss', type=float, default=7.5)
    p.add_argument('--cfg_slat', type=float, default=7.5)

    p.add_argument('--ss_lr', type=float, default=5.0)
    p.add_argument('--slat_lr', type=float, default=0.01)
    p.add_argument('--ss_max_iter', type=int, default=15)
    p.add_argument('--slat_max_iter', type=int, default=15)
    p.add_argument('--ss_use_distribution_prior', action='store_true', help='Enable sparse-noise distribution prior in optinit sparse optimization.')
    p.add_argument('--ss_optimize_in_spatial', action='store_true', help='Optimize sparse noise in spatial domain (disable frequency-domain optimization).')
    p.add_argument('--ss_opt_steps', type=int, default=None, help='Override sparse sampler steps during sparse optimization loop only.')
    p.add_argument('--slat_opt_steps', type=int, default=None, help='Override SLAT sampler steps during SLAT optimization loop only.')
    p.add_argument('--ss_opt_cfg_scale', type=float, default=1.0, help='Scale sparse cfg_strength during sparse optimization loop only.')

    p.add_argument('--blend_strength_ss', type=float, default=1.0)
    p.add_argument('--blend_strength_slat', type=float, default=1.0)

    p.add_argument('--ilvr_strength_ss', type=float, default=0.7)
    p.add_argument('--ilvr_downsample_factor', type=int, default=4)
    p.add_argument('--ilvr_weight', type=float, default=1.0)

    p.add_argument('--sdedit_strength_ss', type=float, default=0.75)
    p.add_argument('--sdedit_strength_slat', type=float, default=0.7)

    p.add_argument('--dps_weight_ss', type=float, default=0.3)
    p.add_argument('--dps_weight_slat', type=float, default=0.3)

    p.add_argument('--formats', nargs='+', choices=['gaussian', 'radiance_field', 'mesh'], default=['gaussian', 'radiance_field', 'mesh'])
    p.add_argument('--run_base_again', action='store_true', help='Regenerate base outputs even if base folder already has required files.')
    p.add_argument('--skip_comparison_views', action='store_true', help='Do not save comparison view sequences (view0_* folders/videos).')
    p.add_argument('--eval_num_points', type=int, default=200000)
    p.add_argument('--eval_tau_list', nargs='+', type=float, default=[0.005, 0.01, 0.02])
    p.add_argument('--with_clip', action='store_true', help='Also compute gen-clipscore from sample_gs_gen.mp4 during main eval.')
    p.add_argument('--skip_eval', action='store_true')
    p.add_argument('--dry_run', action='store_true', help='Only parse metadata and print planned samples.')
    return p.parse_args()


def _resolve_latent_root(root: str, fallback: str) -> str:
    if os.path.isdir(root):
        return root
    if os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError(f'Latent root not found: {root} (fallback: {fallback})')


def _find_latent_file(root: str, model: str, sha256: str) -> str:
    candidates = [
        os.path.join(root, f'{sha256}.pt'),
        os.path.join(root, f'{sha256}.npz'),
        os.path.join(root, model, f'{sha256}.pt'),
        os.path.join(root, model, f'{sha256}.npz'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Latent file not found for {sha256} under {root} (model={model})')


def load_base_z(latent_file: str) -> torch.Tensor:
    if latent_file.endswith('.pt'):
        z = torch.load(latent_file, map_location='cpu')
        if not isinstance(z, torch.Tensor):
            raise ValueError(f'Expected tensor in {latent_file}')
        return z

    data = np.load(latent_file)
    if 'mean' in data:
        z = torch.tensor(data['mean']).float()
    elif 'x_0' in data:
        z = torch.tensor(data['x_0']).float()
    else:
        raise ValueError(f'Unsupported sparse latent format: {latent_file}')
    return z


def ensure_base_z_batch(z: torch.Tensor) -> torch.Tensor:
    # Expected shape for TRELLIS sparse latent: [B, C, D, H, W]
    if z.ndim == 4:
        z = z.unsqueeze(0)
    if z.ndim != 5:
        raise ValueError(f'base_z must be 4D or 5D tensor, got shape={tuple(z.shape)}')
    return z.float().contiguous()


def load_base_slat(latent_file: str, sp) -> object:
    if latent_file.endswith('.pt'):
        slat = torch.load(latent_file, map_location='cpu', weights_only=False)
        if not hasattr(slat, 'coords') or not hasattr(slat, 'feats'):
            raise ValueError(f'Unsupported slat tensor object in {latent_file}')
        coords = slat.coords.int()
        if coords.ndim != 2:
            raise ValueError(f'Invalid slat coords ndim from {latent_file}: {coords.ndim}')
        if coords.shape[1] == 3:
            coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1)
        elif coords.shape[1] != 4:
            raise ValueError(f'Invalid slat coords shape from {latent_file}: {tuple(coords.shape)}')
        return sp.SparseTensor(coords=coords, feats=slat.feats.float())

    data = np.load(latent_file)
    if 'coords' not in data or 'feats' not in data:
        raise ValueError(f'Unsupported slat latent format: {latent_file}')
    coords = torch.tensor(data['coords']).int()
    if coords.ndim != 2:
        raise ValueError(f'Invalid slat coords ndim from {latent_file}: {coords.ndim}')
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords.int()], dim=1)
    elif coords.shape[1] != 4:
        raise ValueError(f'Invalid slat coords shape from {latent_file}: {tuple(coords.shape)}')
    feats = torch.tensor(data['feats']).float()
    return sp.SparseTensor(coords=coords, feats=feats)


def load_mask(mask_path: str) -> torch.Tensor:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    m = torch.load(mask_path).float()
    if m.ndim != 3:
        raise ValueError('Mask must be 3D tensor.')
    return m.unsqueeze(0).unsqueeze(0).cuda()


def parse_captions(captions_field: str) -> List[str]:
    if captions_field is None:
        return []
    v = str(captions_field).strip()
    if v == '' or v.lower() in {'nan', 'none'}:
        return []
    try:
        parsed = ast.literal_eval(v)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for x in parsed:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    return out


def build_prompt_dict(metadata_csv: str) -> Tuple[Dict[str, str], List[str]]:
    prompt_by_sha: Dict[str, str] = {}
    sha_order: List[str] = []
    with open(metadata_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = (row.get('sha256') or '').strip()
            if not sha:
                continue
            captions = parse_captions(row.get('captions'))
            if len(captions) == 0:
                continue
            prompt_by_sha[sha] = captions[0]
            sha_order.append(sha)
    return prompt_by_sha, sha_order


def run_method(
    pipeline,
    method: str,
    prompt: str,
    base_z: torch.Tensor,
    base_slat,
    mask: torch.Tensor,
    args,
):
    common = {
        'seed': args.seed,
        'sparse_structure_sampler_params': {'steps': args.steps, 'cfg_strength': args.cfg_ss},
        'slat_sampler_params': {'steps': args.steps, 'cfg_strength': args.cfg_slat},
        'formats': args.formats,
    }

    t0 = time.perf_counter()

    if method == 'optinit':
        out_list, geo_list, (z_out, slat_out) = pipeline.optinit(
            prompt,
            base_z,
            base_slat,
            mask,
            sparse_structure_optmizer_params={
                'lr': args.ss_lr,
                'max_iter': args.ss_max_iter,
                'verbose': False,
                'use_distribution_prior': args.ss_use_distribution_prior,
                'use_frequency_optimization': (not args.ss_optimize_in_spatial),
                'opt_steps': args.ss_opt_steps,
                'opt_cfg_scale': args.ss_opt_cfg_scale,
            },
            slat_optimizer_params={
                'lr': args.slat_lr,
                'max_iter': args.slat_max_iter,
                'verbose': False,
                'opt_steps': args.slat_opt_steps,
            },
            **common,
        )
    elif method == 'blendeddiffusion':
        out_list, geo_list, (z_out, slat_out) = pipeline.blended_diffusion(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            edit_mask=mask,
            blend_strength_ss=args.blend_strength_ss,
            blend_strength_slat=args.blend_strength_slat,
            return_intermediates=True,
            **common,
        )
    elif method == 'md':
        out_list, geo_list, (z_out, slat_out) = pipeline.multidiffusion(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            edit_mask=mask,
            return_intermediates=True,
            **common,
        )
    elif method == 'ilvr':
        out_list, geo_list, (z_out, slat_out) = pipeline.ilvr(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            edit_mask=mask,
            sparse_structure_strength=args.ilvr_strength_ss,
            downsample_factor=args.ilvr_downsample_factor,
            ilvr_weight=args.ilvr_weight,
            return_intermediates=True,
            **common,
        )
    elif method == 'sdedit':
        out_list, geo_list, (z_out, slat_out) = pipeline.sdedit(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            sdedit_mask=mask,
            sparse_structure_strength=args.sdedit_strength_ss,
            slat_strength=args.sdedit_strength_slat,
            return_intermediates=True,
            **common,
        )
    elif method == 'repaint':
        out_list, geo_list, (z_out, slat_out) = pipeline.repaint(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            repaint_mask=mask,
            return_intermediates=True,
            **common,
        )
    elif method == 'dps':
        out_list, geo_list, (z_out, slat_out) = pipeline.dps(
            prompt,
            base_z=base_z,
            base_slat=base_slat,
            edit_mask=mask,
            dps_weight_ss=args.dps_weight_ss,
            dps_weight_slat=args.dps_weight_slat,
            return_intermediates=True,
            **common,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    runtime = time.perf_counter() - t0
    return out_list, geo_list, z_out, slat_out, runtime


def save_metrics(path: str, metrics: Dict[str, float]):
    with open(os.path.join(path, 'evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        for k in sorted(metrics.keys()):
            f.write(f'{k}: {metrics[k]}\n')
    with open(os.path.join(path, 'evaluation_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)


def base_outputs_ready(base_dir: str, formats: List[str]) -> bool:
    checks = [os.path.join(base_dir, 'base.png')]
    if 'gaussian' in formats:
        checks.extend([
            os.path.join(base_dir, 'sample_gs.mp4'),
            os.path.join(base_dir, 'sample_gs_recon.mp4'),
            os.path.join(base_dir, 'sample_gs_gen.mp4'),
            os.path.join(base_dir, 'dinov2', 'full.npz'),
            os.path.join(base_dir, 'dinov2', 'edit_only.npz'),
        ])
    if 'mesh' in formats:
        checks.extend([
            os.path.join(base_dir, 'sample_mesh.mp4'),
            os.path.join(base_dir, 'sample_mesh_recon.mp4'),
            os.path.join(base_dir, 'sample.glb'),
        ])
    if 'radiance_field' in formats:
        checks.append(os.path.join(base_dir, 'sample_rf.mp4'))
    return all(os.path.exists(p) for p in checks)


def evaluate_outputs(
    base_dir: str,
    out_dir: str,
    mask: torch.Tensor,
    seed: int,
    prompt: str,
    args,
    eval_utils,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    base_glb = os.path.join(base_dir, 'sample.glb')
    edited_glb = os.path.join(out_dir, 'sample.glb')
    if os.path.exists(base_glb) and os.path.exists(edited_glb):
        metrics.update(eval_utils.evaluate_optinit_glb_preserve_region(
            base_glb_path=base_glb,
            edited_glb_path=edited_glb,
            mask=mask,
            num_points=args.eval_num_points,
            tau_list=args.eval_tau_list,
            seed=seed,
        ))

    base_mesh = os.path.join(base_dir, 'sample_mesh_recon.mp4')
    edited_mesh = os.path.join(out_dir, 'sample_mesh_recon.mp4')
    if os.path.exists(base_mesh) and os.path.exists(edited_mesh):
        metrics.update(eval_utils.evaluate_video_psnr_ssim_lpips(
            base_video_path=base_mesh,
            edited_video_path=edited_mesh,
            prefix='geo-',
        ))

    base_gs = os.path.join(base_dir, 'sample_gs_recon.mp4')
    edited_gs = os.path.join(out_dir, 'sample_gs_recon.mp4')
    if os.path.exists(base_gs) and os.path.exists(edited_gs):
        metrics.update(eval_utils.evaluate_video_psnr_ssim_lpips(
            base_video_path=base_gs,
            edited_video_path=edited_gs,
            prefix='app-',
        ))

    edited_gs_gen = os.path.join(out_dir, 'sample_gs_gen.mp4')
    if args.with_clip and prompt and os.path.exists(edited_gs_gen):
        metrics.update(eval_utils.evaluate_clip_video_text(
            video_path=edited_gs_gen,
            text=prompt,
            prefix='gen-',
        ))

    return metrics


def main():
    args = parse_args()

    if args.dataset_name is not None:
        ds = args.dataset_name
        if args.metadata_csv == 'datasets/Toys4k_fullview/metadata.csv':
            args.metadata_csv = os.path.join('datasets', ds, 'metadata.csv')
        if args.ss_latent_root == 'datasets/Toys4k_fullview/ss_latents':
            args.ss_latent_root = os.path.join('datasets', ds, 'ss_latents')
        if args.latent_root == 'datasets/Toys4k_fullview/latents':
            args.latent_root = os.path.join('datasets', ds, 'latents')
    default_ds = args.dataset_name if args.dataset_name else 'Toys4k_fullview'
    args.ss_latent_root = _resolve_latent_root(args.ss_latent_root, os.path.join('datasets', default_ds, 'ss_latents'))
    args.latent_root = _resolve_latent_root(args.latent_root, os.path.join('datasets', default_ds, 'latents'))

    prompt_by_sha, sha_order = build_prompt_dict(args.metadata_csv)
    valid_shas = sha_order

    if args.sha256_file:
        with open(args.sha256_file, 'r', encoding='utf-8') as f:
            wanted = {line.strip() for line in f if line.strip()}
        valid_shas = [sha for sha in valid_shas if sha in wanted]

    if args.sha256:
        valid_shas = [sha for sha in valid_shas if sha == args.sha256]
    else:
        valid_shas = valid_shas[args.offset:]
        if args.limit is not None:
            valid_shas = valid_shas[:args.limit]

    print(f'[info] metadata: {args.metadata_csv}')
    print(f'[info] prompts with non-empty captions: {len(prompt_by_sha)}')
    print(f'[info] planned samples after offset/limit: {len(valid_shas)}')
    print(f'[info] methods: {args.methods}')

    if args.dry_run:
        for sha in valid_shas[:20]:
            print(f'{sha} -> {prompt_by_sha[sha]}')
        return

    import trellis.modules.sparse as sp
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import render_utils
    from trellis.utils import evaluation_utils as eval_utils

    pipeline = TrellisTextTo3DPipeline.from_pretrained('microsoft/TRELLIS-text-xlarge')
    pipeline.cuda()
    mask = load_mask(args.mask)

    num_done = 0
    for sha in valid_shas:
        prompt = prompt_by_sha[sha]

        try:
            z_file = _find_latent_file(args.ss_latent_root, args.ss_latent_model, sha)
            slat_file = _find_latent_file(args.latent_root, args.latent_model, sha)
            base_z = ensure_base_z_batch(load_base_z(z_file))
            base_slat = load_base_slat(slat_file, sp)
        except Exception as e:
            print(f'[skip] {sha}: failed to load base latents ({e})')
            continue

        sample_root = os.path.join('logs', args.tag, sha)
        base_dir = os.path.join(sample_root, 'base')
        os.makedirs(base_dir, exist_ok=True)

        if args.run_base_again or not base_outputs_ready(base_dir, args.formats):
            try:
                with torch.no_grad():
                    base_outputs = pipeline.decode_slat(base_slat.cuda(), args.formats)
                base_outputs = detach_tree(base_outputs)
                render_utils.save_outputs(base_outputs, base_dir, args.formats, preserve_mask=mask)
                if not args.skip_comparison_views:
                    render_utils.save_comparison_view([base_outputs], base_dir)
            except Exception as e:
                print(f'[skip] {sha}: failed to decode/save base outputs ({e})')
                continue
        else:
            print(f'[base-skip] {sha}: base outputs already exist at {base_dir}')

        for method in args.methods:
            out_subdir = method
            if method == 'optinit':
                out_subdir = args.optinit_out_subdir
            out_dir = os.path.join(sample_root, out_subdir)
            os.makedirs(out_dir, exist_ok=True)

            print(f'[run] {sha} | {method} | prompt="{prompt}"')
            try:
                out_list, geo_list, z_out, slat_out, runtime = run_method(
                    pipeline=pipeline,
                    method=method,
                    prompt=prompt,
                    base_z=base_z.cuda(),
                    base_slat=base_slat.cuda(),
                    mask=mask,
                    args=args,
                )
            except Exception as e:
                print(f'[fail] {sha} | {method}: {e}')
                continue

            out_list = detach_tree(out_list)
            geo_list = detach_tree(geo_list)
            z_out = detach_tree(z_out)
            slat_out = detach_tree(slat_out)

            torch.save(z_out.detach().cpu(), os.path.join(out_dir, f'{method}_z.pt'))
            torch.save(slat_out.to('cpu'), os.path.join(out_dir, f'{method}_slat.pt'))
            render_utils.save_outputs(out_list[-1], out_dir, args.formats, preserve_mask=mask)
            if not args.skip_comparison_views:
                render_utils.save_comparison_view(out_list, os.path.join(out_dir, 'view0_optslat'))
                render_utils.save_comparison_view(geo_list, os.path.join(out_dir, 'view0_optss'))

            if args.skip_eval:
                continue

            metrics = {'time_sec': float(runtime)}
            try:
                metrics.update(evaluate_outputs(base_dir, out_dir, mask, args.seed, prompt, args, eval_utils))
            except Exception as e:
                print(f'[warn] {sha} | {method}: evaluation failed ({e})')
            save_metrics(out_dir, metrics)

        num_done += 1
        print(f'[done] {sha} ({num_done}/{len(valid_shas)})')


if __name__ == '__main__':
    main()
