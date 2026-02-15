import os
import argparse
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

parser = argparse.ArgumentParser(description="Trellis Inpainting Example")
parser.add_argument("--base_prompt", type=str, required=True, help="Text prompt for original generation")
parser.add_argument("--inpaint_prompt", type=str, required=True, help="Text prompt for inpainting")
parser.add_argument("--inpaint_mask", type=str, required=True, help="Path to the inpainting mask image")
parser.add_argument("--seed", type=int, default=1, help="Random seed for generation")
parser.add_argument("--steps", type=int, default=12, help="Number of steps for the sampler")
parser.add_argument("--cfg_ss", type=float, default=7.5, help="Classifier-free guidance strength for sparse structure sampler")
parser.add_argument("--cfg_slat", type=float, default=7.5, help="Classifier-free guidance strength for SLAT sampler")
parser.add_argument("--seed_inpaint", type=int, default=1, help="Random seed for inpainting generation")
parser.add_argument("--steps_inpaint", type=int, default=12, help="Number of steps for the inpainting sampler")
parser.add_argument("--cfg_ss_inpaint", type=float, default=7.5, help="Classifier-free guidance strength for sparse structure sampler in inpainting")
parser.add_argument("--cfg_slat_inpaint", type=float, default=7.5, help="Classifier-free guidance strength for SLAT sampler in inpainting")
parser.add_argument("--formats", nargs="+", type=str, choices=["gaussian", "radiance_field", "mesh"], default=['gaussian', 'radiance_field'], help="Formats to generate")
parser.add_argument("--tag", type=str, default="default", help="Additional tag for the output directory")
parser.add_argument("--run_base_again", action='store_true', help="Whether to run the base generation again even if outputs already exist")
parser.add_argument("--verbose", action='store_true', help="Whether to run in debug mode with fewer steps and samples for quick testing")
args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

### Generation for base 3D
basedir = f"logs/{args.tag}/{args.base_prompt.replace(' ', '')}/seed{args.seed}_steps{args.steps}_cfgss{args.cfg_ss}_cfgslat{args.cfg_slat}"

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
    render_utils.save_outputs(base_output, basedir, args.formats)
    render_utils.save_comparison_view([base_output], basedir)



### Inpainting (required: base_z, base_slat, inpaint_prompt, inpaint_mask)
# Load inpainting mask 
if args.inpaint_mask:
    assert os.path.exists(args.inpaint_mask), f"Mask path {args.inpaint_mask} does not exist."
    inpaint_mask = torch.load(args.inpaint_mask).float()  # (H, W, D), values in [0, 1], where 1 indicates the region to be inpainted 
    assert inpaint_mask.ndim == 3, "Inpainting mask must be a 3D tensor."
    inpaint_mask = inpaint_mask.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W, D)
else:
    raise ValueError("Inpainting mask (--inpaint_mask) must be provided.")

# Run inpainting
inpaintdir = os.path.join(basedir, f"inpaint/{args.inpaint_prompt.replace(' ', '')}/seed{args.seed_inpaint}_steps{args.steps_inpaint}_cfgss{args.cfg_ss_inpaint}_cfgslat{args.cfg_slat_inpaint}")
os.makedirs(inpaintdir, exist_ok=True)

inpaint_output, (inpaint_z, inpaint_slat) = pipeline.inpaint(
    args.inpaint_prompt,
    base_z,
    base_slat,
    inpaint_mask,
    seed=args.seed_inpaint,
    sparse_structure_sampler_params={
        "steps": args.steps_inpaint,
        "cfg_strength": args.cfg_ss_inpaint,
    },
    sparse_structure_optmizer_params={
        "lr": 10.0,
        "max_iter": 10,
        "verbose": args.verbose,
    },
    slat_sampler_params={
        "steps": args.steps_inpaint,
        "cfg_strength": args.cfg_slat_inpaint,
    },
    slat_optimizer_params={
        "lr": 0.01,
        "max_iter": 15,
        "verbose": args.verbose,
    },
    formats=args.formats,
)

torch.save(inpaint_z, os.path.join(inpaintdir, f"inpaint_z.pt"))
torch.save(inpaint_slat, os.path.join(inpaintdir, f"inpaint_slat.pt"))
render_utils.save_outputs(inpaint_output[-1], inpaintdir, args.formats)
render_utils.save_comparison_view(inpaint_output, os.path.join(inpaintdir,"view0"))

if 'gaussian' in args.formats:
    render_utils.save_gaussian_through_iter(inpaint_output, os.path.join(inpaintdir,"videos"))

### concatenate video for easy comparison
base_vid = imageio.v3.imread(os.path.join(basedir,"sample_gs.mp4"))
inpaint_vid = imageio.v3.imread(os.path.join(inpaintdir,"sample_gs.mp4"))
imageio.mimwrite(os.path.join(inpaintdir,"video_comparison_gs.mp4"), np.concatenate([base_vid, inpaint_vid], axis=2), fps=30)


### concatenate video in view0 for easy comparison
base_view = imageio.v3.imread(os.path.join(basedir,"view.png"))
out_vid = []
for i in range(len(inpaint_output)):
    inp_view = imageio.v3.imread(os.path.join(inpaintdir,"view0",f"view_slat{i}.png"))
    out_vid.append(np.concatenate((base_view, inp_view), axis=1))
imageio.mimwrite(os.path.join(inpaintdir,"view0_comparison_gs.mp4"), np.stack(out_vid, axis=0), fps=3)

### load
from trellis.utils.data_utils import load_gsply
from trellis.utils.evaluation_utils import compare_geo_color, get_keepidx
base_ply = load_gsply(os.path.join(basedir,"sample.ply"))
inpaint_ply = load_gsply(os.path.join(inpaintdir,"sample.ply"))

base_keep = get_keepidx(base_ply['xyz'], (inpaint_mask[0,0].cpu()<0))
inpaint_keep = get_keepidx(inpaint_ply['xyz'], (inpaint_mask[0,0].cpu()<0))
output = compare_geo_color(base_ply['xyz'][base_keep], base_ply['rgb'][base_keep], inpaint_ply['xyz'][inpaint_keep], inpaint_ply['rgb'][inpaint_keep]) # mask 넣어야됌
print(" / ".join([f"{k}:{v:.4f}" for k, v in output.items()]))


### Evaluation
# import pdb; pdb.set_trace()
# from trellis.utils.evaluation_utils import evaluate_inpainting
# f = open(os.path.join(inpaintdir, "evaluation.txt"), "w")
# for i in range(len(inpaint_output)):  ## 그냥 마지막만 하는게 맞을까? 
#     metrics = evaluate_inpainting(base_slat, inpaint_slat_list[i], base_output, inpaint_output[i], inpaint_mask)
#     metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
#     print(f"Inpainting Evaluation Metrics for iteration {i}, {metrics_str}")
#     f.write(f"Iteration {i}, {metrics_str}\n")
# f.close()