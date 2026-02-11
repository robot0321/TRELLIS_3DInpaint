import os
import argparse
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import pickle
import torch
import imageio
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
parser.add_argument("--steps_inpaint", type=int, default=12, help="Number of steps for the inpainting sampler")
parser.add_argument("--cfg_ss_inpaint", type=float, default=7.5, help="Classifier-free guidance strength for sparse structure sampler in inpainting")
parser.add_argument("--cfg_slat_inpaint", type=float, default=7.5, help="Classifier-free guidance strength for SLAT sampler in inpainting")
parser.add_argument("--format", type=list, default=['gaussian', 'radiance_field', 'mesh'], help="Formats to generate")
parser.add_argument("--tag", type=str, default="default", help="Additional tag for the output directory")
args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

### Generation for base 3D
basedir = f"logs/{args.tag}/{args.base_prompt.replace(' ', '')}/seed{args.seed}_steps{args.steps}_cfgss{args.cfg_ss}_cfgslat{args.cfg_slat}"

if os.path.exists(f"{basedir}/base_z.pt") and os.path.exists(f"{basedir}/base_slat.pt"):
    # Load original 3D asset
    base_z = torch.load(os.path.join(basedir, "base_z.pt"))
    base_slat = torch.load(os.path.join(basedir, "base_slat.pt"))
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
    ### format들 save해서... 비교...
    # import pdb; pdb.set_trace()
    # torch.save(base_output, os.path.join(basedir, "base_output.pt"))
    render_utils.save_outputs(base_output, basedir, args.format)



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
inpaintdir = os.path.join(basedir, f"inpaint/{args.base_prompt.replace(' ', '')}/seed{args.seed}_steps{args.steps_inpaint}_cfgss{args.cfg_ss_inpaint}_cfgslat{args.cfg_slat_inpaint}")
os.makedirs(inpaintdir, exist_ok=True)

inpaint_output, (inpaint_z_list, inpaint_slat_list) = pipeline.inpaint(
    args.inpaint_prompt,
    base_z,
    base_slat,
    inpaint_mask,
    seed=args.seed,
    sparse_structure_sampler_params={
        "steps": args.steps_inpaint,
        "cfg_strength": args.cfg_ss_inpaint,
    },
    sparse_structure_optmizer_params={
        "lr": 3.0,
        "max_iter": 20,
        "verbose": True,
    },
    slat_sampler_params={
        "steps": args.steps_inpaint,
        "cfg_strength": args.cfg_slat_inpaint,
    },
    slat_optimizer_params={
        "lr": 3.0,
        "max_iter": 20,
        "verbose": True,
    },
)
for i in range(len(inpaint_output)): # Save each iteration's result
    os.makedirs(os.path.join(inpaintdir, f"iter{i}"), exist_ok=True)
    torch.save(inpaint_z_list[i], os.path.join(inpaintdir, f"iter{i}/inpaint_z.pt"))
    torch.save(inpaint_slat_list[i], os.path.join(inpaintdir, f"iter{i}/inpaint_slat.pt"))
    ### format들 save해서... 비교...
    render_utils.save_outputs(inpaint_output[i], os.path.join(inpaintdir, f"iter{i}"), args.format)


### Evaluation
import pdb; pdb.set_trace()
from trellis.utils.evaluation_utils import evaluate_inpainting
f = open(os.path.join(inpaintdir, "evaluation.txt"), "w")
for i in range(len(inpaint_output)):  ## 그냥 마지막만 하는게 맞을까? 
    metrics = evaluate_inpainting(base_slat, inpaint_slat_list[i], base_output, inpaint_output[i], inpaint_mask)
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Inpainting Evaluation Metrics for iteration {i}, {metrics_str}")
    f.write(f"Iteration {i}, {metrics_str}\n")
f.close()