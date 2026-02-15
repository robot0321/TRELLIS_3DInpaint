from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..utils.evaluation_utils import compute_iou_and_dice, compare_geo_color, get_keepidx
from ..renderers.sh_utils import SH2RGB

class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords, z_s

    def optimize_initnoise_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        inpaint_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        optmizer_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            base_z (torch.Tensor): The base sparse structure latent.
            inpaint_mask (torch.Tensor): The inpaint mask.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            optmizer_params (dict): Additional parameters for the optimizer.
        """
        
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution 
        noise_original = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        noise = noise_original.clone().float()
        
        coords_list, z_s_list = [], []
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        if base_z is not None and inpaint_mask is not None:
            mask_s = F.interpolate(inpaint_mask, size=(reso, reso, reso), mode='trilinear')
            mask_s = (mask_s < 0.5).float() # flip the mask
            target_z = base_z * mask_s
            target_occ = decoder(target_z)>0
        
        # optimize spectral noise with Adam
        noise_freq = torch.fft.rfftn(noise, s=noise.shape[-3:], dim=(-3, -2, -1))
        noise_optim_target = torch.nn.Parameter(noise_freq.detach().clone(), requires_grad=True)
        optimizer = torch.optim.Adam([noise_optim_target], lr=optmizer_params['lr'], betas=(0.9, 0.999))

        count = mask_s.sum(dim=(-3,-2,-1), keepdim=True)
        max_iter = optmizer_params['max_iter']
        for i in range(max_iter):
            noise_spatial = torch.fft.irfftn(noise_optim_target, s=noise.shape[-3:], dim=(-3, -2, -1))
            noise_combined = noise_spatial * mask_s + noise_original * (1 - mask_s)

            mm = (noise_spatial*mask_s).sum(dim=(-3,-2,-1))/count
            vv = (noise_spatial.pow(2)*mask_s).sum(dim=(-3,-2,-1))/count - mm.pow(2)
            if optmizer_params['verbose']:
                # print(f"{i}-th / cond: ", mm.mean().item(), vv.mean().item())
                pass
            
            with torch.no_grad():
                z_s = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise_combined,
                    **cond,
                    **sampler_params,
                    verbose=False
                ).samples
                added_noise = noise_combined - z_s
                z_s_list.append(z_s)
                
                # Decode occupancy latent
                occ = decoder(z_s*mask_s)>0
                coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()
                coords_list.append(coords)     

            # compute z_s_0_hat with gradients
            z_s_0_hat = noise_combined - added_noise
            
            # masked loss and optimizer step
            loss = F.mse_loss(target_z, z_s_0_hat*mask_s, reduction="none")
            loss = (loss * mask_s).mean()

            ### prior
            gmean = (noise_spatial * mask_s).sum(dim=(-3,-2,-1), keepdim=True) / count
            gstd = torch.sqrt( (mask_s*(noise_spatial-gmean)**2).sum(dim=(-3,-2,-1), keepdim=True) / count + 1e-8 )
            loss = loss + 31.6*(gmean**2).mean() + 10.0*((gstd-1)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if optmizer_params['verbose']:
                iou, dice = compute_iou_and_dice(target_occ, occ)
                print(f"iter{i} / iou: {iou:.3f}, dice: {dice:.3f}, loss: {loss.item():.6f}")
                coords_list.append(coords)

        noise_spatial = torch.fft.irfftn(noise_optim_target, s=noise.shape[-3:], dim=(-3, -2, -1))
        noise_spatial_final = noise_spatial * mask_s + noise_original * (1 - mask_s)
        with torch.no_grad():
            z_s = self.sparse_structure_sampler.sample(flow_model, noise_spatial_final, **cond, **sampler_params, verbose=False).samples
            z_s_list.append(z_s)
            coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
            coords_list.append(coords)
        
        
        return coords_list, z_s_list

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def optimize_initnoise_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        inpaint_mask: torch.Tensor,
        sampler_params: dict = {},
        optimizer_params: dict = {}
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            base_slat (sp.SparseTensor): The base structured latent.
            inpaint_mask (torch.Tensor): The inpainting mask.
            sampler_params (dict): Additional parameters for the sampler.
        """

        
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        slatnorm_std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        slatnorm_mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)
        
        mask_s = (inpaint_mask[0,0] < 0.5) ## fliped mask
        noise_feats_original = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)
        noise_feats = noise_feats_original
        
        ###
        if optimizer_params['verbose']:
            with torch.no_grad():
                target_gs = self.decode_slat(base_slat,['gaussian'])['gaussian'][0]
                target_keep = get_keepidx(target_gs._xyz, mask_s)
                
                
        slat_list = []
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        if base_slat is not None and inpaint_mask is not None:
            # 1) base_slat -> idx_grid (64^3) 1개만
            base_idx_grid = torch.full((reso,reso,reso), -1, dtype=torch.long, device=self.device)
            base_idx_grid[base_slat.coords[:,1], base_slat.coords[:,2], base_slat.coords[:,3]] \
                = torch.arange(base_slat.feats.shape[0], device=self.device, dtype=torch.long)
            
            # 2) (inpaint) coords에서 mask 통과하는 좌표만
            inp_x, inp_y, inp_z = coords[:,1].long(), coords[:,2].long(), coords[:,3].long()
            
            valid = mask_s[inp_x, inp_y, inp_z]
            inp_x, inp_y, inp_z = inp_x[valid], inp_y[valid], inp_z[valid]
            
            # 3) cond에 존재하는지 + idx 뽑기
            target_slat_idx = base_idx_grid[inp_x, inp_y, inp_z]        # (K',)
            keep = target_slat_idx >= 0
            if keep.sum()>0:
                target_slat_idx = target_slat_idx[keep]       # (M,)
                
                # 4) feat gather & 겹치는 좌표
                target_slat_feat = base_slat.feats[target_slat_idx]  # (M,8)
                # matched_coords = torch.stack([qx[keep], qy[keep], qz[keep]], dim=-1)  # (M,3)
    
                noise_feats_spatial = torch.nn.Parameter(noise_feats_original.detach().clone(), requires_grad=True)
                optimizer = torch.optim.Adam([noise_feats_spatial], lr=optimizer_params['lr'])
            
                for i in range(optimizer_params['max_iter']):
                    noise_feats_combined = noise_feats_spatial*valid[:,None] + noise_feats_original*(~valid[:,None])
                    
                    with torch.no_grad():
                        noise = sp.SparseTensor(feats=noise_feats_combined, coords=coords)
                        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=False).samples
                        slat = slat * slatnorm_std + slatnorm_mean
                        
                        added_noise = noise_feats_combined - slat.feats
                        slat_list.append(slat)
                        
                    # compute z_s_0_hat with gradients
                    slatfeat_0_hat = noise_feats_combined - added_noise
                    
                    # masked loss and optimizer step
                    loss = F.mse_loss(target_slat_feat, slatfeat_0_hat[valid][keep], reduction="none")
                    loss = (loss).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if optimizer_params['verbose']:
                        with torch.no_grad():
                            out_gs = self.decode_slat(slat,['gaussian'])['gaussian'][0]
                            out_keep = get_keepidx(out_gs._xyz, mask_s)
                            
                            slat_metric = compare_geo_color(target_gs._xyz[target_keep].detach().cpu().numpy(), 
                                                            SH2RGB(target_gs._features_dc[target_keep]).detach().cpu().numpy(), 
                                                            out_gs._xyz[out_keep].detach().cpu().numpy(),
                                                            SH2RGB(out_gs._features_dc[out_keep]).detach().cpu().numpy()) # inpaint_mask
                            
                        print(f"iter{i}/loss:{loss.item():.6f}/"+"/".join([f"{k}:{v:.4f}" for k, v in slat_metric.items()]))
                        
                        
                
                ### final
                noise_feats = noise_feats_spatial*valid[:,None] + noise_feats_original*(~valid[:,None])
        
        noise = sp.SparseTensor(feats=noise_feats, coords=coords)
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=False).samples
        slat = slat * slatnorm_std + slatnorm_mean
        slat_list.append(slat)
        
        # for i, slat in enumerate(slat_list):
        #     std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        #     mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        #     slat_list[i] = slat * std + mean
        
        
        return slat_list


    @torch.no_grad()
    def run(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords, z_s = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats), (z_s, slat)

    def inpaint(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        inpaint_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        sparse_structure_optmizer_params: dict = {},
        slat_sampler_params: dict = {},
        slat_optimizer_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords_list, z_s_list = self.optimize_initnoise_sparse_structure(cond, base_z, inpaint_mask, num_samples, sparse_structure_sampler_params, sparse_structure_optmizer_params)
        ## 이상적으로 골라진 coords가 있다고 가정 (여기서는 마지막으로 일단...)

        # slat = self.sample_slat(cond, coords_list[-1], slat_sampler_params) 
        # output = self.decode_slat(slat, formats)
        # return [output], (z_s_list[-1], slat)

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        slat_list = self.optimize_initnoise_slat(cond, coords_list[-1], base_slat, inpaint_mask, slat_sampler_params, slat_optimizer_params) 

        output_list = []
        with torch.no_grad():
            for slat in slat_list:
                output_list.append(self.decode_slat(slat, formats))
        
        return output_list, (z_s_list[-1], slat_list[-1])
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
