import os
import argparse
import csv
import time
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import pickle
import torch
import imageio
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.utils.evaluation_utils import (
    evaluate_optinit_glb_preserve_region,
    evaluate_video_psnr_ssim_lpips,
    evaluate_clip_video_text,
)

parser = argparse.ArgumentParser(description="Trellis Optinit Example")
parser.add_argument("--base_prompt", type=str, required=True, help="Text prompt for original generation")
parser.add_argument("--optinit_prompt", type=str, required=True, help="Text prompt for optinit")
parser.add_argument("--optinit_mask", type=str, required=True, help="Path to the optinit mask image")
parser.add_argument("--seed", type=int, default=1, help="Random seed for generation")
parser.add_argument("--steps", type=int, default=12, help="Number of steps for the sampler")
parser.add_argument("--cfg_ss", type=float, default=7.5, help="Classifier-free guidance strength for sparse structure sampler")
parser.add_argument("--cfg_slat", type=float, default=7.5, help="Classifier-free guidance strength for SLAT sampler")
parser.add_argument("--seed_optinit", type=int, default=1, help="Random seed for optinit generation")
parser.add_argument("--steps_optinit", type=int, default=12, help="Number of steps for the optinit sampler")
parser.add_argument("--cfg_ss_optinit", type=float, default=7.5, help="Classifier-free guidance strength for sparse structure sampler in optinit")
parser.add_argument("--cfg_slat_optinit", type=float, default=7.5, help="Classifier-free guidance strength for SLAT sampler in optinit")
parser.add_argument("--ss_lr", type=float, default=3.0, help="Learning rate for optimizing sparse structure during optinit")
parser.add_argument("--slat_lr", type=float, default=0.01, help="Learning rate for optimizing slat during optinit")
parser.add_argument("--ss_max_iter", type=int, default=15, help="Max iterations for optimizing sparse structure during optinit")
parser.add_argument("--slat_max_iter", type=int, default=15, help="Max iterations for optimizing slat during optinit")
parser.add_argument("--ss_use_distribution_prior", action="store_true", help="Enable sparse-noise distribution prior during sparse optimization.")
parser.add_argument("--ss_optimize_in_spatial", action="store_true", help="Optimize sparse noise directly in spatial domain instead of frequency domain.")
parser.add_argument("--ss_opt_steps", type=int, default=None, help="Override sparse sampler steps during sparse optimization loop only.")
parser.add_argument("--slat_opt_steps", type=int, default=None, help="Override SLAT sampler steps during SLAT optimization loop only.")
parser.add_argument("--ss_opt_cfg_scale", type=float, default=1.0, help="Scale sparse cfg_strength during sparse optimization loop only.")
parser.add_argument("--formats", nargs="+", type=str, choices=["gaussian", "radiance_field", "mesh"], default=['gaussian', 'radiance_field', "mesh"], help="Formats to generate")
parser.add_argument("--tag", type=str, default="default", help="Additional tag for the output directory")
parser.add_argument("--exp_name", type=str, default="", help="Optional suffix to separate experiment output folders.")
parser.add_argument("--run_base_again", action='store_true', help="Whether to run the base generation again even if outputs already exist")
parser.add_argument("--verbose", action='store_true', help="Whether to run in debug mode with fewer steps and samples for quick testing")
parser.add_argument("--eval_num_points", type=int, default=200000, help="Surface points for CD/F-score evaluation")
parser.add_argument("--eval_tau_list", nargs="+", type=float, default=[0.005, 0.01, 0.02], help="F-score thresholds")
parser.add_argument("--skip_eval", action="store_true", help="Skip metric evaluation and summary export")
args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

if args.optinit_mask:
    assert os.path.exists(args.optinit_mask), f"Mask path {args.optinit_mask} does not exist."
    optinit_mask = torch.load(args.optinit_mask).float()  # (H, W, D), values in [0, 1]
    assert optinit_mask.ndim == 3, "Optinit mask must be a 3D tensor."
    optinit_mask = optinit_mask.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W, D)
else:
    raise ValueError("Optinit mask (--optinit_mask) must be provided.")

### Generation for base 3D
basedir = f"logs/{args.tag}/{args.base_prompt.replace(' ', '')}/seed{args.seed}_steps{args.steps}_cfgss{args.cfg_ss}_cfgslat{args.cfg_slat}"

base_output = None
if os.path.exists(f"{basedir}/base_z.pt") and os.path.exists(f"{basedir}/base_slat.pt") and not args.run_base_again:
    # Load original 3D asset
    base_z = torch.load(os.path.join(basedir, "base_z.pt"))
    base_slat = torch.load(os.path.join(basedir, "base_slat.pt"), weights_only=False)
    ### load format들... 비교...
    # base_output = torch.load(os.path.join(basedir, "base_output.pt"))
else:
    # Run the pipeline
    os.makedirs(basedir, exist_ok=True)
    base_output, (base_z, base_slat) = pipeline.run(
        args.base_prompt,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg_ss,
        },
        slat_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg_slat,
        },
    )
    torch.save(base_z, os.path.join(basedir, "base_z.pt"))
    torch.save(base_slat, os.path.join(basedir, "base_slat.pt"))
    # import pdb; pdb.set_trace()
    ### format들 save해서... 비교...
    # import pdb; pdb.set_trace()
    # torch.save(base_output, os.path.join(basedir, "base_output.pt"))
    render_utils.save_outputs(base_output, basedir, args.formats, preserve_mask=optinit_mask)
    render_utils.save_comparison_view([base_output], basedir)

base_mesh_recon_path = os.path.join(basedir, "sample_mesh_recon.mp4")
base_glb_path = os.path.join(basedir, "sample.glb")
if ("mesh" in args.formats) and (not os.path.exists(base_mesh_recon_path) or not os.path.exists(base_glb_path)):
    if base_output is None:
        base_output = pipeline.decode_slat(base_slat, args.formats)
    render_utils.save_outputs(base_output, basedir, args.formats, preserve_mask=optinit_mask)
    render_utils.save_comparison_view([base_output], basedir)

### Optinit (required: base_z, base_slat, optinit_prompt, optinit_mask)

# Run optinit
exp_tokens = []
if args.ss_use_distribution_prior:
    exp_tokens.append("ssprior")
if args.ss_optimize_in_spatial:
    exp_tokens.append("ssspatial")
if args.ss_opt_steps is not None:
    exp_tokens.append(f"ssoptsteps{args.ss_opt_steps}")
if args.slat_opt_steps is not None:
    exp_tokens.append(f"slatoptsteps{args.slat_opt_steps}")
if float(args.ss_opt_cfg_scale) != 1.0:
    exp_tokens.append(f"ssoptcfgx{args.ss_opt_cfg_scale:g}")
if args.exp_name:
    exp_tokens.append(args.exp_name)
exp_suffix = ""
if exp_tokens:
    exp_suffix = "_" + "_".join(token.replace(".", "p") for token in exp_tokens)

optinitdir = os.path.join(
    basedir,
    (
        f"optinit/{args.optinit_prompt.replace(' ', '')}/"
        f"seed{args.seed_optinit}_steps{args.steps_optinit}_cfgss{args.cfg_ss_optinit}_cfgslat{args.cfg_slat_optinit}"
        f"{exp_suffix}"
    ),
)
os.makedirs(optinitdir, exist_ok=True)

t0 = time.perf_counter()
optinit_output, optinit_geometry_output, (optinit_z, optinit_slat) = pipeline.optinit(
    args.optinit_prompt,
    base_z,
    base_slat,
    optinit_mask,
    seed=args.seed_optinit,
    sparse_structure_sampler_params={
        "steps": args.steps_optinit,
        "cfg_strength": args.cfg_ss_optinit,
    },
    sparse_structure_optmizer_params={
        "lr": args.ss_lr,
        "max_iter": args.ss_max_iter,
        "verbose": args.verbose,
        "use_distribution_prior": args.ss_use_distribution_prior,
        "use_frequency_optimization": not args.ss_optimize_in_spatial,
        "opt_steps": args.ss_opt_steps,
        "opt_cfg_scale": args.ss_opt_cfg_scale,
    },
    slat_sampler_params={
        "steps": args.steps_optinit,
        "cfg_strength": args.cfg_slat_optinit,
    },
    slat_optimizer_params={
        "lr": args.slat_lr,
        "max_iter": args.slat_max_iter,
        "verbose": args.verbose,
        "opt_steps": args.slat_opt_steps,
    },
    formats=args.formats,
)
optinit_runtime_sec = time.perf_counter() - t0

torch.save(optinit_z, os.path.join(optinitdir, f"optinit_z.pt"))
torch.save(optinit_slat, os.path.join(optinitdir, f"optinit_slat.pt"))
render_utils.save_outputs(optinit_output[-1], optinitdir, args.formats, preserve_mask=optinit_mask)
render_utils.save_comparison_view(optinit_output, os.path.join(optinitdir,"view0_optslat"))
render_utils.save_comparison_view(optinit_geometry_output, os.path.join(optinitdir,"view0_optss"))

if 'gaussian' in args.formats:
    # render_utils.save_gaussian_through_iter(optinit_output, os.path.join(optinitdir,"videos"))
    pass

### concatenate video for easy comparison
base_vid = imageio.v3.imread(os.path.join(basedir,"sample_gs.mp4"))
optinit_vid = imageio.v3.imread(os.path.join(optinitdir,"sample_gs.mp4"))
imageio.mimwrite(os.path.join(optinitdir,"video_comparison_gs.mp4"), np.concatenate([base_vid, optinit_vid], axis=2), fps=30)


### concatenate video in view0 for easy comparison
base_view = imageio.v3.imread(os.path.join(basedir,"base.png"))
optslat_vid = []
for i in range(len(optinit_output)):
    optslat_vid.append(imageio.v3.imread(os.path.join(optinitdir,"view0_optslat",f"iter{i}.png")))
optslat_vid = np.stack(optslat_vid, axis=0)
imageio.mimwrite(os.path.join(optinitdir,"view0_optslat","view0_optslat.mp4"), optslat_vid, fps=args.slat_max_iter/5)
imageio.mimwrite(os.path.join(optinitdir,"view0_optslat_comparison.mp4"), np.concatenate((base_view[None].repeat(len(optslat_vid), axis=0), optslat_vid), axis=2), fps=args.slat_max_iter/5)

# if args.verbose:
optss_vid = []
for i in range(len(optinit_geometry_output)):
    optss_vid.append(imageio.v3.imread(os.path.join(optinitdir,"view0_optss",f"iter{i}.png")))
optss_vid = np.stack(optss_vid, axis=0)
imageio.mimwrite(os.path.join(optinitdir,"view0_optss","view0_optss.mp4"), optss_vid, fps=args.ss_max_iter/5)
imageio.mimwrite(os.path.join(optinitdir,"view0_optss_comparison.mp4"), np.concatenate((base_view[None].repeat(len(optss_vid), axis=0), optss_vid), axis=2), fps=args.ss_max_iter/5)



### load
from trellis.utils.data_utils import load_gsply
from trellis.utils.evaluation_utils import compare_geo_color, get_keepidx
base_ply = load_gsply(os.path.join(basedir,"sample.ply"))
optinit_ply = load_gsply(os.path.join(optinitdir,"sample.ply"))

base_keep = get_keepidx(base_ply['xyz'], (optinit_mask[0,0].cpu()<0))
optinit_keep = get_keepidx(optinit_ply['xyz'], (optinit_mask[0,0].cpu()<0))
output = compare_geo_color(base_ply['xyz'][base_keep], base_ply['rgb'][base_keep], optinit_ply['xyz'][optinit_keep], optinit_ply['rgb'][optinit_keep]) # mask 넣어야됌
print(" / ".join([f"{k}:{v:.4f}" for k, v in output.items()]))

if not args.skip_eval:
    metrics = {"time_sec": float(optinit_runtime_sec)}
    base_glb = os.path.join(basedir, "sample.glb")
    optinit_glb = os.path.join(optinitdir, "sample.glb")
    base_mesh_recon = os.path.join(basedir, "sample_mesh_recon.mp4")
    optinit_mesh_recon = os.path.join(optinitdir, "sample_mesh_recon.mp4")
    base_gs_recon = os.path.join(basedir, "sample_gs_recon.mp4")
    optinit_gs_recon = os.path.join(optinitdir, "sample_gs_recon.mp4")
    optinit_gs_gen = os.path.join(optinitdir, "sample_gs_gen.mp4")

    if os.path.exists(base_glb) and os.path.exists(optinit_glb):
        metrics.update(evaluate_optinit_glb_preserve_region(
            base_glb_path=base_glb,
            edited_glb_path=optinit_glb,
            mask=optinit_mask,
            num_points=args.eval_num_points,
            tau_list=args.eval_tau_list,
            seed=args.seed_optinit,
        ))

    if os.path.exists(base_mesh_recon) and os.path.exists(optinit_mesh_recon):
        metrics.update(evaluate_video_psnr_ssim_lpips(
            base_video_path=base_mesh_recon,
            edited_video_path=optinit_mesh_recon,
            prefix="geo-",
        ))

    if os.path.exists(base_gs_recon) and os.path.exists(optinit_gs_recon):
        metrics.update(evaluate_video_psnr_ssim_lpips(
            base_video_path=base_gs_recon,
            edited_video_path=optinit_gs_recon,
            prefix="app-",
        ))

    if os.path.exists(optinit_gs_gen):
        metrics.update(evaluate_clip_video_text(
            video_path=optinit_gs_gen,
            text=args.optinit_prompt,
            prefix="gen-",
        ))

    if len(metrics) > 0:
        txt_path = os.path.join(optinitdir, "evaluation_metrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for k in sorted(metrics.keys()):
                f.write(f"{k}: {metrics[k]}\n")

        csv_path = os.path.join(optinitdir, "evaluation_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)


### Evaluation
# import pdb; pdb.set_trace()
# from trellis.utils.evaluation_utils import evaluate_optinit
# f = open(os.path.join(optinitdir, "evaluation.txt"), "w")
# for i in range(len(optinit_output)):  ## 그냥 마지막만 하는게 맞을까? 
#     metrics = evaluate_optinit(base_slat, optinit_slat_list[i], base_output, optinit_output[i], optinit_mask)
#     metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
#     print(f"Optinit Evaluation Metrics for iteration {i}, {metrics_str}")
#     f.write(f"Iteration {i}, {metrics_str}\n")
# f.close()
