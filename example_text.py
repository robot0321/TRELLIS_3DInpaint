import argparse
import os
from pathlib import Path
from typing import Optional

os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np
import torch

from trellis.modules import sparse as sp
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


def _make_view_video(view_dir: Path, out_name: str, fps: float) -> None:
    frames = []
    i = 0
    while True:
        p = view_dir / f"iter{i}.png"
        if not p.exists():
            break
        frames.append(imageio.v3.imread(p))
        i += 1
    if len(frames) == 0:
        return
    vid = np.stack(frames, axis=0)
    imageio.mimwrite(view_dir / out_name, vid, fps=max(1.0, float(fps)))


def _single_seed(
    pipeline: TrellisTextTo3DPipeline,
    prompt: str,
    seed: int,
    out_root: Path,
    formats,
    steps: int,
    cfg_ss: float,
    cfg_slat: float,
    jitrate: Optional[float],
    save_view0: bool,
) -> None:
    prompt_tag = prompt.replace(' ', '')
    outdir = out_root / prompt_tag / f"seed{seed}_steps{steps}_cfgss{cfg_ss}_cfgslat{cfg_slat}"
    if jitrate is not None:
        outdir = outdir / f"jitrate{jitrate}"
    outdir.mkdir(parents=True, exist_ok=True)

    sparse_structure_sampler_params = {
        "steps": steps,
        "cfg_strength": cfg_ss,
    }
    slat_sampler_params = {
        "steps": steps,
        "cfg_strength": cfg_slat,
    }

    with torch.no_grad():
        cond = pipeline.get_cond([prompt])
        torch.manual_seed(seed)

        # Keep intermediate states for sparse-structure and SLAT to export view0_ss/view0_slat.
        ss_flow_model = pipeline.models['sparse_structure_flow_model']
        ss_decoder = pipeline.models['sparse_structure_decoder']
        reso = ss_flow_model.resolution
        ss_noise = torch.randn(1, ss_flow_model.in_channels, reso, reso, reso, device=pipeline.device)
        if jitrate is not None:
            ss_noise = ss_noise + float(jitrate) * torch.randn_like(ss_noise)
        ss_params = {**pipeline.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        ss_ret = pipeline.sparse_structure_sampler.sample(
            ss_flow_model,
            ss_noise,
            **cond,
            **ss_params,
            verbose=True,
        )
        z_s = ss_ret.samples
        coords = torch.argwhere(ss_decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        slat_flow_model = pipeline.models['slat_flow_model']
        slat_input = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], slat_flow_model.in_channels, device=pipeline.device),
            coords=coords,
        )
        slat_params = {**pipeline.slat_sampler_params, **slat_sampler_params}
        slat_ret = pipeline.slat_sampler.sample(
            slat_flow_model,
            slat_input,
            **cond,
            **slat_params,
            verbose=True,
        )

        std = torch.tensor(pipeline.slat_normalization['std'])[None].to(slat_ret.samples.device)
        mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(slat_ret.samples.device)
        slat = slat_ret.samples * std + mean

        outputs = pipeline.decode_slat(slat, formats)
        
    render_utils.save_outputs(outputs, str(outdir), formats)

    # glb = postprocessing_utils.to_glb(
    #     outputs['gaussian'][0],
    #     outputs['mesh'][0],
    #     simplify=0.95,
    #     texture_size=1024,
    # )
    # glb.export(outdir / "sample.glb")
    # outputs['gaussian'][0].save_ply(str(outdir / "sample.ply"))

    if not save_view0:
        return
    
    with torch.no_grad():
        # view0_ss: geometry-only progression reconstructed from sparse-structure intermediates.
        z_s_list = [*ss_ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(ss_decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        geo_slat_list = pipeline.init_geo_slat(cond, coords_list, slat_sampler_params)
        geo_output_list = []
        for geo_slat in geo_slat_list:
            if geo_slat is None:
                geo_output_list.append(None)
            else:
                geo_output_list.append(pipeline.decode_slat(geo_slat, ['gaussian']))

        view0_ss_dir = outdir / "view0_ss"
        render_utils.save_comparison_view(geo_output_list, str(view0_ss_dir))
        _make_view_video(view0_ss_dir, "view0_ss.mp4", fps=steps / 5)

        # view0_slat: appearance progression from SLAT denoising intermediates.
        slat_list = [(s * std + mean) for s in [*slat_ret.pred_x_t, slat_ret.samples]]
        slat_output_list = [pipeline.decode_slat(s, ['gaussian']) for s in slat_list]
        view0_slat_dir = outdir / "view0_slat"
        render_utils.save_comparison_view(slat_output_list, str(view0_slat_dir))
        _make_view_video(view0_slat_dir, "view0_slat.mp4", fps=steps / 5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-to-3D generation with optional view0_ss/view0_slat exports and sparse-structure noise jitter"
    )
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', type=int, default=0, help='Start seed.')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of consecutive seeds to run.')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--cfg_ss', type=float, default=7.5)
    parser.add_argument('--cfg_slat', type=float, default=7.5)
    parser.add_argument(
        '--jitrate',
        nargs='+',
        type=float,
        default=None,
        help='If set, run once per value and apply ss_noise += jitrate * Gaussian noise. Outputs are saved per jitrate folder.',
    )
    parser.add_argument('--outdir', type=str, default='logs_figure')
    parser.add_argument(
        '--formats',
        nargs='+',
        type=str,
        choices=['gaussian', 'radiance_field', 'mesh'],
        default=['gaussian', 'radiance_field', 'mesh'],
    )
    parser.add_argument('--save_view0', action='store_true', help='Save view0_ss / view0_slat images and videos.')
    args = parser.parse_args()

    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    pipeline.cuda()

    jitrates = args.jitrate if args.jitrate is not None else [None]

    for seed in range(args.seed, args.seed + args.num_seeds):
        for jitrate in jitrates:
            print(f"[run] prompt='{args.prompt}' seed={seed} jitrate={jitrate}")
            _single_seed(
                pipeline=pipeline,
                prompt=args.prompt,
                seed=seed,
                out_root=Path(args.outdir),
                formats=args.formats,
                steps=args.steps,
                cfg_ss=args.cfg_ss,
                cfg_slat=args.cfg_slat,
                jitrate=jitrate,
                save_view0=args.save_view0,
            )


if __name__ == '__main__':
    main()
