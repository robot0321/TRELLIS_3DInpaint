import os
import imageio
import glob
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_prompt", type=str, required=True, help="Text prompt for original generation")
parser.add_argument("--inpaint_prompt", type=str, required=True, help="Text prompt for inpainting")
parser.add_argument("--logpath", type=str, default="logs/test_draft1", help="Path to the logs directory")
parser.add_argument("--mode", type=str, choices=["rotate", "optss", "optslat"], default="rotate", help="Modes to include in the video")
args = parser.parse_args()


base_prompt=args.base_prompt.replace(' ', '')
inpaint_prompt=args.inpaint_prompt.replace(' ', '')
logpath=args.logpath
mode = args.mode
# base_prompt="Treesonthegrass"
# inpaint_prompt="Aredroofhouse"
# inpaint_prompt="Snowyhousesinwinter"

###
pathlist = sorted(glob.glob(os.path.join(logpath, base_prompt, "seed*")))
base_videos = []
for path in tqdm(pathlist):
    inp_videos = []
    if mode=="rotate":
        base_vid = imageio.v3.imread(os.path.join(path, 'sample_gs.mp4'))
    elif mode=="optss" or mode=="optslat":
        base_vid = imageio.v3.imread(os.path.join(path, 'base.png'))
    inp_videos.append(base_vid)
    
    inpaint_pathlist = sorted(glob.glob(os.path.join(path,f'inpaint/{inpaint_prompt}/seed*')))
    for inp_path in inpaint_pathlist:
        if mode=="rotate":
            inp_vid = imageio.v3.imread(os.path.join(inp_path, 'sample_gs.mp4'))
        elif mode=="optss":
            inp_vid = imageio.v3.imread(os.path.join(inp_path, 'view0_optss', 'view0_optss.mp4'))
        elif mode=="optslat":
            inp_vid = imageio.v3.imread(os.path.join(inp_path, 'view0_optslat', 'view0_optslat.mp4'))
        else:
            raise ValueError(f"Invalid mode: {mode}")
        inp_videos.append(inp_vid)

    if mode=="optss" or mode=="optslat":
        # repeat the base view to match the number of inpaint videos
        inp_videos[0] = np.repeat(inp_videos[0][None], len(inp_videos[1]), axis=0)

    inp_videos = np.concatenate(inp_videos, axis=2)
    base_videos.append(inp_videos)

base_videos = np.concatenate(base_videos, axis=1)
if mode=="rotate":
    fps = 30
elif mode=="optss" or mode=="optslat":
    fps = 2

imageio.mimwrite(os.path.join(logpath, base_prompt, f"video_{inpaint_prompt}_{mode}.mp4"), base_videos, fps=fps)