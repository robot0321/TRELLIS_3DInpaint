import os
import imageio
import glob
import numpy as np
from tqdm import tqdm

base_prompt="Treesonthegrass"
inpaint_prompt="Snowyhousesinwinter"#"Aredroofhouse"

logpath = "logs/test_draft"
pathlist = sorted(glob.glob(os.path.join(logpath, base_prompt, "seed*")))

base_videos = []
for path in tqdm(pathlist):
    inp_videos = []
    base_vid = imageio.v3.imread(os.path.join(path, 'sample_gs.mp4'))
    inp_videos.append(base_vid)
    
    inpaint_pathlist = sorted(glob.glob(os.path.join(path,f'inpaint/{inpaint_prompt}/seed*')))
    for inp_path in inpaint_pathlist:
        inp_vid = imageio.v3.imread(os.path.join(inp_path, 'sample_gs.mp4'))
        inp_videos.append(inp_vid)
        
    inp_videos = np.concatenate(inp_videos, axis=2)
    base_videos.append(inp_videos)

base_videos = np.concatenate(base_videos, axis=1)
imageio.mimwrite(os.path.join(logpath, base_prompt, f"video_{inpaint_prompt}.mp4"), base_videos, fps=30)