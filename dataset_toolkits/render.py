import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence, upper_hemisphere_hammersley_sequence, hemisphere_fibonacci


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(
    file_path,
    sha256,
    output_dir,
    num_views,
    ishemisphere=False,
    black_pixel_ratio_threshold=0.2,
    black_rgb_threshold=0.02,
    max_black_rerender=2,
    fail_on_black_pixels=False,
    view_distribution="hammersley",
):
    output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    if view_distribution == "hemisphere_fibonacci":
        for i in range(num_views):
            y, p = hemisphere_fibonacci(i, num_views)
            yaws.append(y)
            pitchs.append(p)
    else:
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            if ishemisphere:
                if p > 0:
                    yaws.append(y)
                    pitchs.append(p)
            else: # default
                yaws.append(y)
                pitchs.append(p)
    num_views = len(pitchs)
            
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if True:# rerender_on_black_pixels:
        args.extend([
            '--rerender_on_black_pixels',
            '--black_pixel_ratio_threshold', str(black_pixel_ratio_threshold),
            '--black_rgb_threshold', str(black_rgb_threshold),
            '--max_black_rerender', str(max_black_rerender),
        ])
    if fail_on_black_pixels:
        args.append('--fail_on_black_pixels')
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    parser.add_argument('--hemisphere', action='store_true')
    parser.add_argument('--view_distribution', type=str, default='hammersley',
                        choices=['hammersley', 'hemisphere_fibonacci'],
                        help='Camera sampling distribution.')
    # parser.add_argument('--rerender_on_black_pixels', action='store_true',
    #                     help='Re-render when foreground black pixel ratio is too high.')
    parser.add_argument('--black_pixel_ratio_threshold', type=float, default=0.2,
                        help='Foreground black ratio threshold to trigger re-render.')
    parser.add_argument('--black_rgb_threshold', type=float, default=0.02,
                        help='Foreground pixel is treated as black when max(R,G,B) <= this value.')
    parser.add_argument('--max_black_rerender', type=int, default=2,
                        help='Maximum number of extra render attempts per view.')
    parser.add_argument('--fail_on_black_pixels', action='store_true',
                        help='Fail object render if threshold is still exceeded after retries.')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(
        _render,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        ishemisphere=opt.hemisphere,
        black_pixel_ratio_threshold=opt.black_pixel_ratio_threshold,
        black_rgb_threshold=opt.black_rgb_threshold,
        max_black_rerender=opt.max_black_rerender,
        fail_on_black_pixels=opt.fail_on_black_pixels,
        view_distribution=opt.view_distribution,
    )
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
