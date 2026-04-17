import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import threading
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue as ThreadQueue

import trellis.models as models
import trellis.modules.sparse as sp


torch.set_grad_enabled(False)


def parse_gpu_ids(gpus_arg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for latent encoding, but no GPU was detected.')
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


def get_latent_name(opt):
    if opt.enc_model is None:
        return f'{opt.feat_model}_{opt.enc_pretrained.split("/")[-1]}'
    return f'{opt.feat_model}_{opt.enc_model}_{opt.ckpt}'


def build_encoder(opt, device):
    if opt.enc_model is None:
        encoder = models.from_pretrained(opt.enc_pretrained).eval().to(device)
    else:
        latent_name = get_latent_name(opt)
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).to(device)
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    return encoder


def build_sparse_input(patchtokens, indices):
    return sp.SparseTensor(
        feats=torch.from_numpy(patchtokens).float(),
        coords=torch.cat([
            torch.zeros((patchtokens.shape[0], 1), dtype=torch.int32),
            torch.from_numpy(indices).int(),
        ], dim=1),
    )


def encode_batch(items, encoder, latent_name, output_dir, device):
    sparse_inputs = [
        build_sparse_input(item['patchtokens'], item['indices'])
        for item in items
    ]
    feats = sp.sparse_cat(sparse_inputs, dim=0).to(device)
    latent = encoder(feats, sample_posterior=False)
    if not torch.isfinite(latent.feats).all():
        raise ValueError('Non-finite latent')

    for batch_idx, item in enumerate(items):
        sample_latent = latent[batch_idx]
        pack = {
            'feats': sample_latent.feats.cpu().numpy().astype(np.float32),
            'coords': sample_latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
        }
        save_path = os.path.join(output_dir, 'latents', latent_name, f"{item['sha256']}.npz")
        np.savez_compressed(save_path, **pack)


def gpu_worker(gpu_id, opt_dict, latent_name, task_queue, result_queue):
    opt = edict(opt_dict)
    device = torch.device(f'cuda:{gpu_id}')
    try:
        torch.cuda.set_device(gpu_id)
        encoder = build_encoder(opt, device)
    except Exception as e:
        result_queue.put({'type': 'worker_error', 'gpu': gpu_id, 'error': str(e)})
        result_queue.put({'type': 'worker_done', 'gpu': gpu_id})
        return

    prefetch_queue = ThreadQueue(maxsize=opt.prefetch_size)

    def producer():
        with ThreadPoolExecutor(max_workers=opt.loader_workers) as loader_executor:
            futures = []

            def loader(sha256):
                try:
                    feat_path = os.path.join(opt.output_dir, 'features', opt.feat_model, f'{sha256}.npz')
                    with np.load(feat_path) as feats:
                        patchtokens = feats['patchtokens']
                        indices = feats['indices']
                    prefetch_queue.put({
                        'done': False,
                        'sha256': sha256,
                        'ok': True,
                        'patchtokens': patchtokens,
                        'indices': indices,
                    })
                except Exception as e:
                    prefetch_queue.put({'done': False, 'sha256': sha256, 'ok': False, 'error': str(e)})

            while True:
                sha256 = task_queue.get()
                if sha256 is None:
                    break
                futures.append(loader_executor.submit(loader, sha256))

            for future in futures:
                future.result()

        prefetch_queue.put({'done': True})

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    pending_items = []

    def flush_pending():
        nonlocal pending_items
        if len(pending_items) == 0:
            return
        batch_items = pending_items
        pending_items = []
        try:
            encode_batch(batch_items, encoder, latent_name, opt.output_dir, device)
            for batch_item in batch_items:
                result_queue.put({'type': 'result', 'sha256': batch_item['sha256'], 'ok': True})
        except Exception as e:
            for batch_item in batch_items:
                result_queue.put({'type': 'result', 'sha256': batch_item['sha256'], 'ok': False, 'error': str(e)})

    while True:
        item = prefetch_queue.get()
        if item.get('done'):
            flush_pending()
            break
        sha256 = item['sha256']
        if not item['ok']:
            result_queue.put({'type': 'result', 'sha256': sha256, 'ok': False, 'error': item['error']})
            continue
        pending_items.append(item)
        if len(pending_items) >= opt.batch_size:
            flush_pending()

    producer_thread.join()
    result_queue.put({'type': 'worker_done', 'gpu': gpu_id})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature model')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of objects to encode together on each worker')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids (e.g., 0,1,2) or "all". Default: 0')
    parser.add_argument('--workers_per_gpu', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Number of worker processes per GPU (1-4)')
    parser.add_argument('--load_workers', dest='loader_workers', type=int, default=8,
                        help='Number of CPU loader threads per worker process')
    parser.add_argument('--loader_workers', type=int, default=8,
                        help='Number of CPU loader threads per worker process')
    parser.add_argument('--prefetch_size', type=int, default=4,
                        help='Number of preloaded feature tensors kept in memory per worker process')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    latent_name = get_latent_name(opt)
    os.makedirs(os.path.join(opt.output_dir, 'latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata[f'feature_{opt.feat_model}'] == True]
        if f'latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{latent_name}'] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'latent_{latent_name}': True})
            sha256s.remove(sha256)

    if len(sha256s) == 0:
        records = pd.DataFrame.from_records(records)
        records.to_csv(os.path.join(opt.output_dir, f'latent_{latent_name}_{opt.rank}.csv'), index=False)
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
                args=(gpu_id, opt_dict, latent_name, task_queue, result_queue),
            )
            p.start()
            workers.append(p)

    for sha256 in sha256s:
        task_queue.put(sha256)
    for _ in workers:
        task_queue.put(None)

    finished_workers = 0
    with tqdm(total=len(sha256s), desc='Extracting latents', dynamic_ncols=True) as pbar:
        while finished_workers < len(workers):
            try:
                msg = result_queue.get(timeout=5)
            except Empty:
                continue

            if msg['type'] == 'worker_error':
                print(f"[worker_error] gpu={msg['gpu']}: {msg['error']}")
            elif msg['type'] == 'worker_done':
                finished_workers += 1
            elif msg['type'] == 'result':
                pbar.update(1)
                if msg['ok']:
                    records.append({'sha256': msg['sha256'], f'latent_{latent_name}': True})
                else:
                    print(f"[skip] {msg['sha256']}: encode/save failed ({msg['error']})")

    for p in workers:
        p.join()

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'latent_{latent_name}_{opt.rank}.csv'), index=False)
