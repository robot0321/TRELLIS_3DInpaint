import os
import copy
import sys
import json
import importlib
import argparse
import threading
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue as ThreadQueue
from torchvision import transforms
from PIL import Image


torch.set_grad_enabled(False)


def is_cuda_oom_error(exc):
    return isinstance(exc, RuntimeError) and 'CUDA out of memory' in str(exc)


def get_data(output_dir, frames, sha256, loader_workers):
    with ThreadPoolExecutor(max_workers=loader_workers) as executor:
        def worker(view):
            image_path = os.path.join(output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data


def parse_gpu_ids(gpus_arg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for feature extraction, but no GPU was detected.')
    n_gpu = torch.cuda.device_count()
    if gpus_arg is None:
        return [0]
    if gpus_arg.strip().lower() == 'all':
        return list(range(n_gpu))
    gpu_ids = [int(g.strip()) for g in gpus_arg.split(',') if g.strip() != '']
    if len(gpu_ids) == 0:
        raise ValueError('--gpus is empty. Use e.g. --gpus 0,1 or --gpus all')
    for gid in gpu_ids:
        if gid < 0 or gid >= n_gpu:
            raise ValueError(f'GPU id {gid} is out of range. Available ids: 0..{n_gpu - 1}')
    return gpu_ids


def load_single_object(sha256, opt, transform):
    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
        metadata = json.load(f)
    frames = metadata['frames']
    data = []
    for datum in get_data(opt.output_dir, frames, sha256, opt.loader_workers):
        datum['image'] = transform(datum['image'])
        data.append(datum)
    if len(data) == 0:
        raise ValueError('No valid rendered views were loaded.')
    positions = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
    return data, positions


def process_loaded_object(sha256, data, positions, opt, feature_name, model, n_patch, device, position_chunk_size=None):
    positions = torch.from_numpy(positions).float().to(device)
    indices = ((positions + 0.5) * 64).long()
    if not (torch.all(indices >= 0) and torch.all(indices < 64)):
        raise ValueError('Some vertices are out of bounds.')

    if position_chunk_size is None:
        position_chunk_size = opt.position_chunk_size

    n_views = len(data)
    pack = {
        'indices': indices.cpu().numpy().astype(np.uint8),
    }
    n_positions = positions.shape[0]
    feature_sum = torch.zeros((n_positions, 1024), dtype=torch.float32, device=device)
    for i in range(0, n_views, opt.batch_size):
        batch_data = data[i:i + opt.batch_size]
        bs = len(batch_data)
        batch_images = torch.stack([d['image'] for d in batch_data]).to(device)
        batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).to(device)
        batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).to(device)
        features = model(batch_images, is_training=True)
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        patchtokens = features['x_prenorm'][:, model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
        for j in range(0, n_positions, position_chunk_size):
            uv_chunk = uv[:, j:j + position_chunk_size]
            sampled = F.grid_sample(
                patchtokens,
                uv_chunk.unsqueeze(1),
                mode='bilinear',
                align_corners=False,
            ).squeeze(2).permute(0, 2, 1)
            feature_sum[j:j + position_chunk_size] += sampled.sum(dim=0)
        del batch_images, batch_extrinsics, batch_intrinsics, features, uv, patchtokens

    pack['patchtokens'] = (feature_sum / n_views).cpu().numpy().astype(np.float16)
    save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
    np.savez_compressed(save_path, **pack)


def gpu_worker(gpu_id, opt_dict, feature_name, task_queue, result_queue):
    opt = edict(opt_dict)
    device = torch.device(f'cuda:{gpu_id}')
    try:
        torch.cuda.set_device(gpu_id)
        model = torch.hub.load('facebookresearch/dinov2', opt.model)
        model.eval().to(device)
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        n_patch = 518 // 14
    except Exception as e:
        result_queue.put({'type': 'worker_error', 'gpu': gpu_id, 'error': str(e)})
        result_queue.put({'type': 'worker_done', 'gpu': gpu_id})
        return

    prefetch_queue = ThreadQueue(maxsize=opt.prefetch_size)

    def producer():
        while True:
            sha256 = task_queue.get()
            if sha256 is None:
                break
            try:
                data, positions = load_single_object(sha256, opt, transform)
                prefetch_queue.put({'done': False, 'sha256': sha256, 'ok': True, 'data': data, 'positions': positions})
            except Exception as e:
                prefetch_queue.put({'done': False, 'sha256': sha256, 'ok': False, 'error': str(e)})
        prefetch_queue.put({'done': True})

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    while True:
        item = prefetch_queue.get()
        if item.get('done'):
            break
        sha256 = item['sha256']
        if not item['ok']:
            result_queue.put({'type': 'result', 'sha256': sha256, 'ok': False, 'error': item['error']})
            continue
        try:
            process_loaded_object(sha256, item['data'], item['positions'], opt, feature_name, model, n_patch, device)
            result_queue.put({'type': 'result', 'sha256': sha256, 'ok': True})
        except Exception as e:
            if is_cuda_oom_error(e) and opt.oom_retry_position_chunk_size < opt.position_chunk_size:
                torch.cuda.empty_cache()
                try:
                    process_loaded_object(
                        sha256, item['data'], item['positions'], opt, feature_name, model, n_patch, device,
                        position_chunk_size=opt.oom_retry_position_chunk_size,
                    )
                    result_queue.put({
                        'type': 'result',
                        'sha256': sha256,
                        'ok': True,
                        'message': f'retried with position_chunk_size={opt.oom_retry_position_chunk_size} after OOM',
                    })
                except Exception as retry_e:
                    result_queue.put({'type': 'result', 'sha256': sha256, 'ok': False, 'error': str(retry_e)})
            else:
                result_queue.put({'type': 'result', 'sha256': sha256, 'ok': False, 'error': str(e)})

    producer_thread.join()

    result_queue.put({'type': 'worker_done', 'gpu': gpu_id})
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids (e.g., 0,1,2) or "all". Default: 0')
    parser.add_argument('--workers_per_gpu', type=int, default=2, choices=[1, 2, 3, 4],
                        help='Number of worker processes per GPU (1-4)')
    parser.add_argument('--loader_workers', type=int, default=8,
                        help='Number of CPU loader threads per worker process')
    parser.add_argument('--prefetch_size', type=int, default=2,
                        help='Number of preloaded objects kept in memory per worker process')
    parser.add_argument('--position_chunk_size', type=int, default=16384,
                        help='Number of voxel positions to sample at once during grid sampling')
    parser.add_argument('--oom_retry_position_chunk_size', type=int, default=8192,
                        help='Fallback chunk size used only when a sample hits CUDA OOM')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'feature_{feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{feature_name}'] == False]
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{feature_name}' : True})
            sha256s.remove(sha256)

    if len(sha256s) == 0:
        records = pd.DataFrame.from_records(records)
        records.to_csv(os.path.join(opt.output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)
        sys.exit(0)

    gpu_ids = parse_gpu_ids(opt.gpus)
    workers_per_gpu = opt.workers_per_gpu
    total_workers = len(gpu_ids) * workers_per_gpu
    print(f'Using GPUs: {gpu_ids}')
    print(f'Workers per GPU: {workers_per_gpu} (total workers: {total_workers})')
    print(f'Number of pending objects: {len(sha256s)}')

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    task_queue = mp.SimpleQueue()
    result_queue = mp.Queue()

    workers = []
    opt_dict = dict(opt)
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, opt_dict, feature_name, task_queue, result_queue),
            )
            p.start()
            workers.append(p)

    for sha256 in sha256s:
        task_queue.put(sha256)
    for _ in range(total_workers):
        task_queue.put(None)

    done_workers = 0
    processed = 0
    with tqdm(total=len(sha256s), desc='Extracting features', dynamic_ncols=True) as pbar:
        while done_workers < total_workers:
            try:
                msg = result_queue.get(timeout=5)
            except Empty:
                alive_workers = sum(1 for p in workers if p.is_alive())
                remaining = len(sha256s) - processed
                print(f'[status] processed={processed}/{len(sha256s)} remaining={remaining} alive_workers={alive_workers}')
                continue
            if msg['type'] == 'result':
                processed += 1
                pbar.update(1)
                if msg['ok']:
                    records.append({'sha256': msg['sha256'], f'feature_{feature_name}': True})
                else:
                    print(f"Error processing {msg['sha256']}: {msg['error']}")
            elif msg['type'] == 'worker_error':
                print(f"Worker on GPU {msg['gpu']} failed to initialize: {msg['error']}")
            elif msg['type'] == 'worker_done':
                done_workers += 1

    for p in workers:
        p.join()
        if p.exitcode != 0:
            print(f'Worker process exited with code {p.exitcode}')
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)
        
