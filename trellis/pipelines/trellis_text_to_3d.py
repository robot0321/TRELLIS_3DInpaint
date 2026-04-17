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
        tokenizer = AutoTokenizer.from_pretrained(name, clean_up_tokenization_spaces=False)
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

    def sample_sparse_structure_list(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample sparse structures and return denoising-step intermediates.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.

        Returns:
            Tuple containing final coords/z_s and per-step coords/z_s lists.
        """
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        sample_ret = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_s_list = [*sample_ret.pred_x_t, sample_ret.samples]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords_list[-1], z_s_list[-1], coords_list, z_s_list

    def repaint_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        repaint_mask: torch.Tensor,
        num_samples: int = 1,
        init_noise: Optional[torch.Tensor] = None,
        sampler_params: dict = {},
        return_intermediates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]]:
        """
        Repaint sparse structure latent while preserving non-repaint region from base_z.

        Args:
            cond (dict): The conditioning information.
            base_z (torch.Tensor): The base sparse structure latent.
            repaint_mask (torch.Tensor): Binary repaint mask (1: repaint, 0: preserve).
            num_samples (int): Number of samples.
            sampler_params (dict): Additional parameters for sampler.
        """
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution

        if repaint_mask.ndim == 3:
            repaint_mask = repaint_mask.unsqueeze(0).unsqueeze(0)
        if repaint_mask.ndim != 5:
            raise ValueError("repaint_mask must have shape (H,W,D) or (B,1,H,W,D).")

        repaint_mask = repaint_mask.float().to(self.device)
        mask_s = F.interpolate(repaint_mask, size=(reso, reso, reso), mode='trilinear') > 0.5
        if mask_s.shape[0] == 1 and num_samples > 1:
            mask_s = mask_s.expand(num_samples, -1, -1, -1, -1)

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)
        if base_z.shape[0] != num_samples:
            raise ValueError(f"base_z batch ({base_z.shape[0]}) != num_samples ({num_samples}).")

        if init_noise is None:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso, device=self.device)
        else:
            noise = init_noise.to(self.device)
            if noise.ndim != 5:
                raise ValueError("init_noise for sparse repaint must be rank-5 (B,C,H,W,D).")
            if noise.shape[0] == 1 and num_samples > 1:
                noise = noise.expand(num_samples, -1, -1, -1, -1)
            if noise.shape[0] != num_samples:
                raise ValueError(f"init_noise batch ({noise.shape[0]}) != num_samples ({num_samples}).")
            if noise.shape[1] != flow_model.in_channels or noise.shape[-3:] != (reso, reso, reso):
                raise ValueError(
                    f"init_noise shape mismatch. expected (*,{flow_model.in_channels},{reso},{reso},{reso}), got {tuple(noise.shape)}"
                )
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        repaint_ret = self.sparse_structure_sampler.repaint(
            flow_model,
            noise=noise,
            x0=base_z,
            repaint_mask=mask_s,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_s = repaint_ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s

        z_s_list = [*repaint_ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(z_iter > 0)[:, [0, 2, 3, 4]].int() for z_iter in map(decoder, z_s_list)]
        return coords, z_s, coords_list, z_s_list

    def optimize_optinit_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        optinit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        optmizer_params: dict = {},
        return_final_noise: bool = False,
    ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            base_z (torch.Tensor): The base sparse structure latent.
            optinit_mask (torch.Tensor): The optinit mask.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            optmizer_params (dict): Additional parameters for the optimizer.
                Supported optional keys:
                  - use_distribution_prior (bool): Add Gaussian-moment prior regularization.
                  - use_frequency_optimization (bool): Optimize noise in frequency domain (default: True).
                  - opt_steps (int): Override sampler steps during optimization loop only.
                  - opt_cfg_scale (float): Scale cfg_strength during optimization loop only.
        """
        if base_z is None or optinit_mask is None:
            raise ValueError("optinit sparse optimization requires both base_z and optinit_mask.")
        if 'lr' not in optmizer_params or 'max_iter' not in optmizer_params:
            raise ValueError("optmizer_params must include 'lr' and 'max_iter' for optinit sparse optimization.")
        
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution 
        noise_original = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        noise = noise_original.clone().float()
        best_noise_spatial = noise.clone().float()
        
        coords_list, z_s_list = [], []
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        # Optimization-phase sampler params can differ from final sampling params.
        opt_sampler_params = dict(sampler_params)
        if optmizer_params.get("opt_steps", None) is not None:
            opt_steps = int(optmizer_params["opt_steps"])
            if opt_steps <= 0:
                raise ValueError(f"opt_steps must be positive, got {opt_steps}")
            opt_sampler_params["steps"] = opt_steps

        opt_cfg_scale = float(optmizer_params.get("opt_cfg_scale", 1.0))
        if opt_cfg_scale <= 0:
            raise ValueError(f"opt_cfg_scale must be positive, got {opt_cfg_scale}")
        if opt_cfg_scale != 1.0:
            if "cfg_strength" not in opt_sampler_params:
                raise ValueError("cfg_strength is required in sampler_params when using opt_cfg_scale != 1.0")
            opt_sampler_params["cfg_strength"] = float(opt_sampler_params["cfg_strength"]) * opt_cfg_scale

        final_sampler_params = dict(sampler_params)
        # final_sampler_params["steps"] = 12

        use_distribution_prior = bool(optmizer_params.get("use_distribution_prior", False))
        use_frequency_optimization = bool(optmizer_params.get("use_frequency_optimization", True))
        
        mask_s = F.interpolate(optinit_mask, size=(reso, reso, reso), mode='trilinear')
        mask_s = (mask_s < 0.5).float() # flip the mask
        cfg_mask_s = mask_s.bool()
        target_z = base_z * mask_s
        target_occ = decoder(target_z)>0
        
        ### optimize noise with Adam
        if use_frequency_optimization:
            noise_init = torch.fft.rfftn(noise, s=noise.shape[-3:], dim=(-3, -2, -1))
        else:
            noise_init = noise.detach().clone()
        noise_optim_target = torch.nn.Parameter(noise_init, requires_grad=True)
        optimizer = torch.optim.Adam([noise_optim_target], lr=optmizer_params['lr'], betas=(0.9, 0.999))

        count = mask_s.sum(dim=(-3,-2,-1), keepdim=True)
        if torch.any(count <= 0):
            raise ValueError("optinit_mask has no valid preserve-region voxels after resize for sparse optimization.")
        max_iter = optmizer_params['max_iter']
        max_iou = 0
        for i in range(max_iter):
            if use_frequency_optimization:
                noise_spatial = torch.fft.irfftn(noise_optim_target, s=noise.shape[-3:], dim=(-3, -2, -1))
            else:
                noise_spatial = noise_optim_target
            noise_combined = noise_spatial * mask_s + noise_original * (1 - mask_s)

            # mm = (noise_spatial*mask_s).sum(dim=(-3,-2,-1))/count
            # vv = (noise_spatial.pow(2)*mask_s).sum(dim=(-3,-2,-1))/count - mm.pow(2)
            # if optmizer_params['verbose']: 
            #     # print(f"{i}-th / cond: {mm.mean().item():.4f}, {vv.mean().item():.4f}")
            #     pass
            
            with torch.no_grad():
                z_s = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise_combined,
                    **cond,
                    **opt_sampler_params,
                    # optinit_mask=cfg_mask_s, ## for masked cfg
                    verbose=False
                ).samples
                added_noise = noise_combined - z_s
                z_s_list.append(z_s)
                
                # Decode occupancy latent
                occ = decoder(z_s*mask_s)>0
                iou, dice = compute_iou_and_dice(target_occ, occ)

                if iou > max_iou:
                    max_iou = iou
                    best_noise_spatial = noise_spatial.clone().detach()

                # if optmizer_params['verbose']:
                coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
                coords_list.append(coords)     

            # compute z_s_0_hat with gradients
            z_s_0_hat = noise_combined - added_noise
            
            # masked loss and optimizer step
            loss = F.mse_loss(target_z, z_s_0_hat*mask_s, reduction="none")
            loss = (loss * mask_s).mean()

            ### prior
            # gmean = (noise_spatial * mask_s).sum(dim=(-3,-2,-1), keepdim=True) / count
            # gstd = torch.sqrt( (mask_s*(noise_spatial-gmean)**2).sum(dim=(-3,-2,-1), keepdim=True) / count + 1e-8 )
            # loss = loss + 31.6*(gmean**2).mean() + 10.0*((gstd-1)**2).mean()
            gmean = (noise_spatial).mean(dim=(-3,-2,-1), keepdim=True)
            gstd = torch.sqrt( torch.mean((noise_spatial-gmean)**2, dim=(-3,-2,-1), keepdim=True)+ 1e-8 )
            gskew = (((noise_spatial-gmean)/gstd)**3).mean(dim=(-3,-2,-1), keepdim=True)
            gkurt = (((noise_spatial-gmean)/gstd)**4).mean(dim=(-3,-2,-1), keepdim=True)
            if use_distribution_prior:
                loss = loss + 1.0 * torch.mean(gmean ** 2) + 1.0 * torch.mean((gstd - 1) ** 2) + 0.1 * torch.mean(gskew ** 2) + 0.1 * torch.mean((gkurt - 3.0) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if optmizer_params['verbose']:
                print(f"iter{i} / iou: {iou:.3f}, dice: {dice:.3f}, loss: {loss.item():.6f}, dist:({gmean.mean().item():.4f},{gstd.mean().item():.4f},{gskew.mean().item():.4f},{gkurt.mean().item():.4f})")

        # noise_spatial = torch.fft.irfftn(noise_optim_target, s=noise.shape[-3:], dim=(-3, -2, -1))
        # noise_spatial_final = noise_spatial * mask_s + noise_original * (1 - mask_s)
        ### use the best noise_spatial during optimization
        noise_spatial_final = best_noise_spatial * mask_s + noise_original * (1 - mask_s)
        with torch.no_grad():
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise_spatial_final,
                **cond,
                **final_sampler_params,
                # optinit_mask=cfg_mask_s,
                verbose=False
            ).samples
            z_s_list.append(z_s)
            coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
            coords_list.append(coords)
        
        
        if not return_final_noise:
            return coords_list, z_s_list
        return coords_list, z_s_list, noise_spatial_final.detach()

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

    def sample_slat_list(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> List[sp.SparseTensor]:
        """
        Sample structured latent and return denoising-step intermediates.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.

        Returns:
            List[sp.SparseTensor]: Denormalized SLAT list for each denoising step
                (including the final sample).
        """
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat_ret = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )

        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)
        return [(slat_iter * std + mean) for slat_iter in [*slat_ret.pred_x_t, slat_ret.samples]]

    def decode_slat_list(
        self,
        slat_list: List[sp.SparseTensor],
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> List[dict]:
        """
        Decode a list of structured latents.

        Args:
            slat_list (List[sp.SparseTensor]): SLAT list to decode.
            formats (List[str]): The formats to decode each SLAT to.
        """
        return [self.decode_slat(slat, formats) for slat in slat_list]

    def repaint_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        repaint_mask: torch.Tensor,
        init_noise: Optional[Union[sp.SparseTensor, Dict[str, torch.Tensor]]] = None,
        sampler_params: dict = {},
        return_intermediates: bool = False,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, List[sp.SparseTensor]]]:
        """
        Repaint SLAT while preserving non-repaint region features from base_slat.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): Coordinates sampled from sparse structure.
            base_slat (sp.SparseTensor): Base structured latent.
            repaint_mask (torch.Tensor): Binary repaint mask (1: repaint, 0: preserve).
            sampler_params (dict): Additional parameters for sampler.
        """
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution

        if repaint_mask.ndim == 5:
            repaint_mask = repaint_mask[0, 0]
        elif repaint_mask.ndim != 3:
            raise ValueError("repaint_mask must have shape (H,W,D) or (B,1,H,W,D).")
        mask_s = (repaint_mask.float().to(self.device) > 0.5)

        base_slat = base_slat.to(self.device)
        base_idx_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
        base_idx_grid[base_slat.coords[:, 1], base_slat.coords[:, 2], base_slat.coords[:, 3]] = torch.arange(
            base_slat.feats.shape[0], device=self.device, dtype=torch.long
        )

        qx, qy, qz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        repaint_region = mask_s[qx, qy, qz]
        target_idx = base_idx_grid[qx, qy, qz]

        if init_noise is None:
            noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, device=self.device)
        elif isinstance(init_noise, sp.SparseTensor):
            init_noise = init_noise.to(self.device)
            init_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
            init_grid[init_noise.coords[:, 1], init_noise.coords[:, 2], init_noise.coords[:, 3]] = torch.arange(
                init_noise.feats.shape[0], device=self.device, dtype=torch.long
            )
            mapped_idx = init_grid[qx, qy, qz]
            noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, device=self.device)
            valid_init = mapped_idx >= 0
            if valid_init.any():
                noise_feats[valid_init] = init_noise.feats[mapped_idx[valid_init]]
        elif isinstance(init_noise, dict):
            if "coords" not in init_noise or "feats" not in init_noise:
                raise ValueError("slat init_noise dict must have keys: coords, feats")
            init_coords = init_noise["coords"].to(self.device)
            init_feats = init_noise["feats"].to(self.device)
            init_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
            init_grid[init_coords[:, 1], init_coords[:, 2], init_coords[:, 3]] = torch.arange(
                init_feats.shape[0], device=self.device, dtype=torch.long
            )
            mapped_idx = init_grid[qx, qy, qz]
            noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, device=self.device)
            valid_init = mapped_idx >= 0
            if valid_init.any():
                noise_feats[valid_init] = init_feats[mapped_idx[valid_init]]
        else:
            raise ValueError("init_noise for slat repaint must be None, SparseTensor, or dict(coords, feats).")

        noise = sp.SparseTensor(feats=noise_feats, coords=coords)
        x0_feats = noise.feats.clone()
        has_base = target_idx >= 0
        preserve_idx = (~repaint_region) & has_base
        if preserve_idx.any():
            x0_feats[preserve_idx] = base_slat.feats[target_idx[preserve_idx]]
        x0 = noise.replace(x0_feats)

        # If preserve-region voxel is not present in base_slat, treat it as repaint region.
        repaint_region = repaint_region | (~has_base)
        repaint_feat_mask = repaint_region[:, None]

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        repaint_ret = self.slat_sampler.repaint(
            flow_model,
            noise=noise,
            x0=x0,
            repaint_mask=repaint_feat_mask,
            **cond,
            **sampler_params,
            verbose=True
        )
        slat = repaint_ret.samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if not return_intermediates:
            return slat

        slat_list = []
        for slat_iter in [*repaint_ret.pred_x_t, repaint_ret.samples]:
            slat_list.append(slat_iter * std + mean)
        return slat, slat_list

    def blended_diffusion_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        blend_strength: float = 1.0,
        return_intermediates: bool = False,
    ):
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution

        if edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
        edit_mask = edit_mask.float().to(self.device)
        mask_s = F.interpolate(edit_mask, size=(reso, reso, reso), mode='trilinear').clamp(0, 1)
        if mask_s.shape[0] == 1 and num_samples > 1:
            mask_s = mask_s.expand(num_samples, -1, -1, -1, -1)

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)

        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        ret = self.sparse_structure_sampler.blended_diffusion(
            flow_model,
            noise=noise,
            x0=base_z,
            blend_mask=mask_s,
            blend_strength=blend_strength,
            **cond,
            **sampler_params,
            verbose=True,
        )
        z_s = ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s
        z_s_list = [*ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords, z_s, coords_list, z_s_list

    def blended_diffusion_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        sampler_params: dict = {},
        blend_strength: float = 1.0,
        return_intermediates: bool = False,
    ):
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        if edit_mask.ndim == 5:
            edit_mask = edit_mask[0, 0]
        mask_s = edit_mask.float().to(self.device).clamp(0, 1)

        base_slat = base_slat.to(self.device)
        base_idx_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
        base_idx_grid[base_slat.coords[:, 1], base_slat.coords[:, 2], base_slat.coords[:, 3]] = torch.arange(
            base_slat.feats.shape[0], device=self.device, dtype=torch.long
        )
        qx, qy, qz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        target_idx = base_idx_grid[qx, qy, qz]
        has_base = target_idx >= 0
        edit_region = mask_s[qx, qy, qz]

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        x0_feats = noise.feats.clone()
        if has_base.any():
            x0_feats[has_base] = base_slat.feats[target_idx[has_base]]
        x0 = noise.replace(x0_feats)

        # If base voxel is missing, treat as editable.
        edit_region = torch.where(has_base, edit_region, torch.ones_like(edit_region))
        blend_feat_mask = edit_region[:, None]

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        ret = self.slat_sampler.blended_diffusion(
            flow_model,
            noise=noise,
            x0=x0,
            blend_mask=blend_feat_mask,
            blend_strength=blend_strength,
            **cond,
            **sampler_params,
            verbose=True,
        )
        slat = ret.samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if not return_intermediates:
            return slat
        slat_list = [(s * std + mean) for s in [*ret.pred_x_t, ret.samples]]
        return slat, slat_list

    def ilvr_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        strength: float = 0.6,
        downsample_factor: int = 4,
        ilvr_weight: float = 1.0,
        return_intermediates: bool = False,
    ):
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution

        if edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
        edit_mask = edit_mask.float().to(self.device)
        mask_s = F.interpolate(edit_mask, size=(reso, reso, reso), mode='trilinear') > 0.5
        if mask_s.shape[0] == 1 and num_samples > 1:
            mask_s = mask_s.expand(num_samples, -1, -1, -1, -1)

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        ret = self.sparse_structure_sampler.ilvr(
            flow_model,
            x0=base_z,
            edit_mask=mask_s,
            ref_x0=base_z,
            strength=strength,
            downsample_factor=downsample_factor,
            ilvr_weight=ilvr_weight,
            **cond,
            **sampler_params,
            verbose=True,
        )
        z_s = ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s
        z_s_list = [*ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords, z_s, coords_list, z_s_list

    def dps_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        dps_weight: float = 0.3,
        return_intermediates: bool = False,
    ):
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution

        if edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
        edit_mask = edit_mask.float().to(self.device)
        mask_s = F.interpolate(edit_mask, size=(reso, reso, reso), mode='trilinear').clamp(0, 1)
        if mask_s.shape[0] == 1 and num_samples > 1:
            mask_s = mask_s.expand(num_samples, -1, -1, -1, -1)

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)

        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        ret = self.sparse_structure_sampler.dps(
            flow_model,
            noise=noise,
            x0=base_z,
            dps_mask=mask_s,
            dps_weight=dps_weight,
            **cond,
            **sampler_params,
            verbose=True,
        )
        z_s = ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s
        z_s_list = [*ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords, z_s, coords_list, z_s_list

    def dps_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        sampler_params: dict = {},
        dps_weight: float = 0.3,
        return_intermediates: bool = False,
    ):
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        if edit_mask.ndim == 5:
            edit_mask = edit_mask[0, 0]
        mask_s = edit_mask.float().to(self.device).clamp(0, 1)

        base_slat = base_slat.to(self.device)
        base_idx_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
        base_idx_grid[base_slat.coords[:, 1], base_slat.coords[:, 2], base_slat.coords[:, 3]] = torch.arange(
            base_slat.feats.shape[0], device=self.device, dtype=torch.long
        )
        qx, qy, qz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        target_idx = base_idx_grid[qx, qy, qz]
        has_base = target_idx >= 0
        edit_region = mask_s[qx, qy, qz]

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        x0_feats = noise.feats.clone()
        if has_base.any():
            x0_feats[has_base] = base_slat.feats[target_idx[has_base]]
        x0 = noise.replace(x0_feats)
        edit_region = torch.where(has_base, edit_region, torch.ones_like(edit_region))
        dps_feat_mask = edit_region[:, None]

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        ret = self.slat_sampler.dps(
            flow_model,
            noise=noise,
            x0=x0,
            dps_mask=dps_feat_mask,
            dps_weight=dps_weight,
            **cond,
            **sampler_params,
            verbose=True,
        )
        slat = ret.samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if not return_intermediates:
            return slat
        slat_list = [(s * std + mean) for s in [*ret.pred_x_t, ret.samples]]
        return slat, slat_list

    def multidiffusion_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        return_intermediates: bool = False,
    ):
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution
        if edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
        edit_mask = edit_mask.float().to(self.device)
        mask_main = F.interpolate(edit_mask, size=(reso, reso, reso), mode='trilinear').clamp(0, 1)
        mask_dilate = (F.avg_pool3d(mask_main, kernel_size=3, stride=1, padding=1) > 0).float()
        if mask_main.shape[0] == 1 and num_samples > 1:
            mask_main = mask_main.expand(num_samples, -1, -1, -1, -1)
            mask_dilate = mask_dilate.expand(num_samples, -1, -1, -1, -1)

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        ret = self.sparse_structure_sampler.multidiffusion(
            flow_model,
            noise=noise,
            x0=base_z,
            mask_list=[mask_main, mask_dilate],
            weights=[1.0, 0.5],
            **cond,
            **sampler_params,
            verbose=True,
        )
        z_s = ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s
        z_s_list = [*ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords, z_s, coords_list, z_s_list

    def multidiffusion_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        sampler_params: dict = {},
        return_intermediates: bool = False,
    ):
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        if edit_mask.ndim == 5:
            edit_mask = edit_mask[0, 0]
        mask_main = edit_mask.float().to(self.device).clamp(0, 1)
        mask_dilate = (F.avg_pool3d(mask_main[None, None], kernel_size=3, stride=1, padding=1)[0, 0] > 0).float()

        base_slat = base_slat.to(self.device)
        base_idx_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
        base_idx_grid[base_slat.coords[:, 1], base_slat.coords[:, 2], base_slat.coords[:, 3]] = torch.arange(
            base_slat.feats.shape[0], device=self.device, dtype=torch.long
        )
        qx, qy, qz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        target_idx = base_idx_grid[qx, qy, qz]
        has_base = target_idx >= 0

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        x0_feats = noise.feats.clone()
        if has_base.any():
            x0_feats[has_base] = base_slat.feats[target_idx[has_base]]
        x0 = noise.replace(x0_feats)

        m1 = mask_main[qx, qy, qz]
        m2 = mask_dilate[qx, qy, qz]
        m1 = torch.where(has_base, m1, torch.ones_like(m1))
        m2 = torch.where(has_base, m2, torch.ones_like(m2))
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        ret = self.slat_sampler.multidiffusion(
            flow_model,
            noise=noise,
            x0=x0,
            mask_list=[m1[:, None], m2[:, None]],
            weights=[1.0, 0.5],
            **cond,
            **sampler_params,
            verbose=True,
        )
        slat = ret.samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if not return_intermediates:
            return slat
        slat_list = [(s * std + mean) for s in [*ret.pred_x_t, ret.samples]]
        return slat, slat_list

    def sdedit_sparse_structure(
        self,
        cond: dict,
        base_z: torch.Tensor,
        sdedit_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        strength: float = 0.6,
        return_intermediates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]]:
        """
        SDEdit sparse structure latent from base_z.
        """
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution

        base_z = base_z.to(self.device)
        if base_z.shape[0] == 1 and num_samples > 1:
            base_z = base_z.repeat(num_samples, 1, 1, 1, 1)
        if base_z.shape[0] != num_samples:
            raise ValueError(f"base_z batch ({base_z.shape[0]}) != num_samples ({num_samples}).")
        if sdedit_mask.ndim == 3:
            sdedit_mask = sdedit_mask.unsqueeze(0).unsqueeze(0)
        if sdedit_mask.ndim != 5:
            raise ValueError("sdedit_mask must have shape (H,W,D) or (B,1,H,W,D).")
        sdedit_mask = sdedit_mask.float().to(self.device)
        mask_s = F.interpolate(sdedit_mask, size=(reso, reso, reso), mode='trilinear') > 0.5
        if mask_s.shape[0] == 1 and num_samples > 1:
            mask_s = mask_s.expand(num_samples, -1, -1, -1, -1)

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        sdedit_ret = self.sparse_structure_sampler.sdedit(
            flow_model,
            x0=base_z,
            edit_mask=mask_s,
            ref_x0=base_z,
            **cond,
            **sampler_params,
            strength=strength,
            verbose=True
        )
        z_s = sdedit_ret.samples
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if not return_intermediates:
            return coords, z_s

        z_s_list = [*sdedit_ret.pred_x_t, z_s]
        coords_list = [torch.argwhere(decoder(z_iter) > 0)[:, [0, 2, 3, 4]].int() for z_iter in z_s_list]
        return coords, z_s, coords_list, z_s_list

    def sdedit_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        sdedit_mask: torch.Tensor,
        sampler_params: dict = {},
        strength: float = 0.6,
        return_intermediates: bool = False,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, List[sp.SparseTensor]]]:
        """
        SDEdit SLAT from base_slat projected onto current coords.
        """
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        if sdedit_mask.ndim == 5:
            sdedit_mask = sdedit_mask[0, 0]
        elif sdedit_mask.ndim != 3:
            raise ValueError("sdedit_mask must have shape (H,W,D) or (B,1,H,W,D).")
        mask_s = (sdedit_mask.float().to(self.device) > 0.5)

        base_slat = base_slat.to(self.device)
        base_idx_grid = torch.full((reso, reso, reso), -1, dtype=torch.long, device=self.device)
        base_idx_grid[base_slat.coords[:, 1], base_slat.coords[:, 2], base_slat.coords[:, 3]] = torch.arange(
            base_slat.feats.shape[0], device=self.device, dtype=torch.long
        )

        qx, qy, qz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        target_idx = base_idx_grid[qx, qy, qz]
        has_base = target_idx >= 0
        edit_region = mask_s[qx, qy, qz] | (~has_base)

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        x0_feats = noise.feats.clone()
        if has_base.any():
            x0_feats[has_base] = base_slat.feats[target_idx[has_base]]
        x0 = noise.replace(x0_feats)

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        sdedit_ret = self.slat_sampler.sdedit(
            flow_model,
            x0=x0,
            edit_mask=edit_region[:, None],
            ref_x0=x0,
            **cond,
            **sampler_params,
            strength=strength,
            verbose=True
        )
        slat = sdedit_ret.samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if not return_intermediates:
            return slat

        slat_list = []
        for slat_iter in [*sdedit_ret.pred_x_t, sdedit_ret.samples]:
            slat_list.append(slat_iter * std + mean)
        return slat, slat_list

    def init_geo_slat(
        self,
        cond: dict,
        coords_list: List[torch.Tensor],
        sampler_params: dict = {},
    ) -> List[Optional[sp.SparseTensor]]:
        """
        Initialize the geo_slat for visualization.
        
        Args:
            cond (dict): The conditioning information.
            coords_list (List[torch.Tensor]): The list of coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        
        geo_slat_list: List[Optional[sp.SparseTensor]] = []
        last_valid_coords = None
        for coords in coords_list:
            if coords.numel() == 0:
                if last_valid_coords is None:
                    # Keep timeline length; this step will be rendered as a blank frame.
                    geo_slat_list.append(None)
                    continue
                coords = last_valid_coords
            else:
                last_valid_coords = coords

            noise = sp.SparseTensor(
                feats=torch.zeros((coords.shape[0], flow_model.in_channels), device=self.device),
                coords=coords,
            )
            sampler_params = {**self.slat_sampler_params, **sampler_params}
            slat = self.slat_sampler.sample(flow_model,noise,**cond,**sampler_params,verbose=False).samples

            std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
            slat = slat * std + mean
            geo_slat_list.append(slat)
        
        return geo_slat_list

    def optimize_optinit_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        base_slat: sp.SparseTensor,
        optinit_mask: torch.Tensor,
        sampler_params: dict = {},
        optimizer_params: dict = {},
        return_final_noise: bool = False,
    ) -> Union[List[sp.SparseTensor], Tuple[List[sp.SparseTensor], Dict[str, torch.Tensor]]]:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            base_slat (sp.SparseTensor): The base structured latent.
            optinit_mask (torch.Tensor): The optinit mask.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if base_slat is None or optinit_mask is None:
            raise ValueError("optinit slat optimization requires both base_slat and optinit_mask.")
        if 'lr' not in optimizer_params or 'max_iter' not in optimizer_params:
            raise ValueError("optimizer_params must include 'lr' and 'max_iter' for optinit slat optimization.")

        
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        reso = flow_model.resolution
        slatnorm_std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        slatnorm_mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)
        
        mask_s = (optinit_mask[0,0] < 0.5) ## fliped mask
        noise_feats_original = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)
        noise_feats = noise_feats_original
        
        ###
        if optimizer_params['verbose']:
            with torch.no_grad():
                target_gs = self.decode_slat(base_slat,['gaussian'])['gaussian'][0]
                target_keep = get_keepidx(target_gs._xyz, mask_s)
                
                
        slat_list = []
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        opt_sampler_params = dict(sampler_params)
        if optimizer_params.get("opt_steps", None) is not None:
            opt_steps = int(optimizer_params["opt_steps"])
            if opt_steps <= 0:
                raise ValueError(f"opt_steps must be positive, got {opt_steps}")
            opt_sampler_params["steps"] = opt_steps
        final_sampler_params = dict(sampler_params)
        # final_sampler_params["steps"] = 12
        
        if base_slat is not None and optinit_mask is not None:
            # 1) base_slat -> idx_grid (64^3) 1개만
            base_idx_grid = torch.full((reso,reso,reso), -1, dtype=torch.long, device=self.device)
            base_idx_grid[base_slat.coords[:,1], base_slat.coords[:,2], base_slat.coords[:,3]] \
                = torch.arange(base_slat.feats.shape[0], device=self.device, dtype=torch.long)
            
            # 2) (optinit) coords에서 mask 통과하는 좌표만
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
                        slat = self.slat_sampler.sample(flow_model, noise, **cond, **opt_sampler_params, verbose=False).samples
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
                                                            SH2RGB(out_gs._features_dc[out_keep]).detach().cpu().numpy()) # optinit_mask
                            
                        print(f"iter{i}/loss:{loss.item():.6f}/"+"/".join([f"{k}:{v:.4f}" for k, v in slat_metric.items()]))
                        
                        
                ### final
                noise_feats = noise_feats_spatial*valid[:,None] + noise_feats_original*(~valid[:,None])
        
        noise = sp.SparseTensor(feats=noise_feats, coords=coords)
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **final_sampler_params, verbose=False).samples
        slat = slat * slatnorm_std + slatnorm_mean
        slat_list.append(slat)
        
        # for i, slat in enumerate(slat_list):
        #     std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        #     mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        #     slat_list[i] = slat * std + mean

        if not return_final_noise:
            return slat_list
        return slat_list, {"coords": coords.detach(), "feats": noise_feats.detach()}


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

    @torch.no_grad()
    def run_list(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_geometry_outputs: bool = False,
    ) -> Union[
        Tuple[List[dict], Tuple[torch.Tensor, sp.SparseTensor]],
        Tuple[List[dict], List[Optional[dict]], Tuple[torch.Tensor, sp.SparseTensor]],
    ]:
        """
        Run the pipeline and return step-wise decoded outputs from SLAT denoising.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            return_geometry_outputs (bool): If True, also return sparse-structure
                proxy gaussian outputs for each sparse denoising step.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords, z_s, coords_list, _ = self.sample_sparse_structure_list(
            cond,
            num_samples,
            sparse_structure_sampler_params,
        )
        slat_list = self.sample_slat_list(cond, coords, slat_sampler_params)
        if len(slat_list) == 0:
            raise RuntimeError("sample_slat_list returned an empty list.")
        output_list = self.decode_slat_list(slat_list, formats)
        if not return_geometry_outputs:
            return output_list, (z_s, slat_list[-1])

        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        geo_output_list: List[Optional[dict]] = []
        for geo_slat in geo_slat_list:
            if geo_slat is None:
                geo_output_list.append(None)
            else:
                geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat_list[-1])

    @torch.no_grad()
    def repaint(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        repaint_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_init_noise: Optional[torch.Tensor] = None,
        slat_init_noise: Optional[Union[sp.SparseTensor, Dict[str, torch.Tensor]]] = None,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ) -> dict:
        """
        Repaint 3D asset from base latents with a binary 3D mask.

        Args:
            prompt (str): Text prompt for repainting.
            base_z (torch.Tensor): Base sparse structure latent.
            base_slat (sp.SparseTensor): Base structured latent.
            repaint_mask (torch.Tensor): Binary repaint mask (1: repaint, 0: preserve).
            num_samples (int): Number of samples.
            seed (int): Random seed.
            sparse_structure_sampler_params (dict): Sparse structure sampler params.
            slat_sampler_params (dict): SLAT sampler params.
            formats (List[str]): Output formats.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        repaint_sparse_ret = self.repaint_sparse_structure(
            cond,
            base_z,
            repaint_mask,
            num_samples,
            sparse_init_noise,
            sparse_structure_sampler_params,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = repaint_sparse_ret
        else:
            coords, z_s = repaint_sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        repaint_slat_ret = self.repaint_slat(
            cond,
            coords,
            base_slat,
            repaint_mask,
            slat_init_noise,
            slat_sampler_params,
            return_intermediates=return_intermediates,
        )

        if not return_intermediates:
            slat = repaint_slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = repaint_slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    @torch.no_grad()
    def sdedit(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        sdedit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        sparse_structure_strength: float = 0.6,
        slat_strength: float = 0.6,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ) -> dict:
        """
        Run SDEdit from base latents.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        sdedit_sparse_ret = self.sdedit_sparse_structure(
            cond,
            base_z,
            sdedit_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
            strength=sparse_structure_strength,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = sdedit_sparse_ret
        else:
            coords, z_s = sdedit_sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        sdedit_slat_ret = self.sdedit_slat(
            cond,
            coords,
            base_slat,
            sdedit_mask,
            sampler_params=slat_sampler_params,
            strength=slat_strength,
            return_intermediates=return_intermediates,
        )

        if not return_intermediates:
            slat = sdedit_slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = sdedit_slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    @torch.no_grad()
    def blended_diffusion(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        blend_strength_ss: float = 1.0,
        blend_strength_slat: float = 1.0,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ):
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        sparse_ret = self.blended_diffusion_sparse_structure(
            cond,
            base_z,
            edit_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
            blend_strength=blend_strength_ss,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = sparse_ret
        else:
            coords, z_s = sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        slat_ret = self.blended_diffusion_slat(
            cond,
            coords,
            base_slat,
            edit_mask,
            sampler_params=slat_sampler_params,
            blend_strength=blend_strength_slat,
            return_intermediates=return_intermediates,
        )
        if not return_intermediates:
            slat = slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    @torch.no_grad()
    def ilvr(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        sparse_structure_strength: float = 0.6,
        downsample_factor: int = 4,
        ilvr_weight: float = 1.0,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ):
        """
        ILVR baseline.
        Note: low-frequency constraint is applied only to dense sparse-structure latent.
        SLAT is inpainted with preserve-anchor repaint because irregular sparse coords do not
        support stable ILVR-style low-pass projection.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        sparse_ret = self.ilvr_sparse_structure(
            cond,
            base_z,
            edit_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
            strength=sparse_structure_strength,
            downsample_factor=downsample_factor,
            ilvr_weight=ilvr_weight,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = sparse_ret
        else:
            coords, z_s = sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        slat_ret = self.repaint_slat(
            cond,
            coords,
            base_slat,
            edit_mask,
            sampler_params=slat_sampler_params,
            return_intermediates=return_intermediates,
        )
        if not return_intermediates:
            slat = slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    @torch.no_grad()
    def dps(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        dps_weight_ss: float = 0.3,
        dps_weight_slat: float = 0.3,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ):
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        sparse_ret = self.dps_sparse_structure(
            cond,
            base_z,
            edit_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
            dps_weight=dps_weight_ss,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = sparse_ret
        else:
            coords, z_s = sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        slat_ret = self.dps_slat(
            cond,
            coords,
            base_slat,
            edit_mask,
            sampler_params=slat_sampler_params,
            dps_weight=dps_weight_slat,
            return_intermediates=return_intermediates,
        )
        if not return_intermediates:
            slat = slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    @torch.no_grad()
    def multidiffusion(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        edit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_intermediates: bool = False,
    ):
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        sparse_ret = self.multidiffusion_sparse_structure(
            cond,
            base_z,
            edit_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            coords, z_s, coords_list, _ = sparse_ret
        else:
            coords, z_s = sparse_ret

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        slat_ret = self.multidiffusion_slat(
            cond,
            coords,
            base_slat,
            edit_mask,
            sampler_params=slat_sampler_params,
            return_intermediates=return_intermediates,
        )
        if not return_intermediates:
            slat = slat_ret
            return self.decode_slat(slat, formats), (z_s, slat)

        slat, slat_list = slat_ret
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)
        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat_iter in slat_list:
                output_list.append(self.decode_slat(slat_iter, formats))
            for geo_slat in geo_slat_list:
                if geo_slat is None:
                    geo_output_list.append(None)
                else:
                    geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))
        return output_list, geo_output_list, (z_s, slat)

    def optinit(
        self,
        prompt: str,
        base_z: torch.Tensor,
        base_slat: sp.SparseTensor,
        optinit_mask: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        sparse_structure_optmizer_params: dict = {},
        slat_sampler_params: dict = {},
        slat_optimizer_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        return_optinit_noises: bool = False,
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
        optinit_sparse_ret = self.optimize_optinit_sparse_structure(
            cond,
            base_z,
            optinit_mask,
            num_samples,
            sparse_structure_sampler_params,
            sparse_structure_optmizer_params,
            return_final_noise=return_optinit_noises,
        )
        if return_optinit_noises:
            coords_list, z_s_list, sparse_init_noise = optinit_sparse_ret
        else:
            coords_list, z_s_list = optinit_sparse_ret
        ## 이상적으로 골라진 coords가 있다고 가정 (여기서는 마지막으로 일단...)

        # slat = self.sample_slat(cond, coords_list[-1], slat_sampler_params) 
        # output = self.decode_slat(slat, formats)
        # return [output], (z_s_list[-1], slat)

        base_slat = sp.SparseTensor(feats=base_slat.feats, coords=base_slat.coords)
        optinit_slat_ret = self.optimize_optinit_slat(
            cond,
            coords_list[-1],
            base_slat,
            optinit_mask,
            slat_sampler_params,
            slat_optimizer_params,
            return_final_noise=return_optinit_noises,
        )
        if return_optinit_noises:
            slat_list, slat_init_noise = optinit_slat_ret
        else:
            slat_list = optinit_slat_ret
        # if sparse_structure_optmizer_params['verbose']:
        geo_slat_list = self.init_geo_slat(cond, coords_list, slat_sampler_params)

        output_list, geo_output_list = [], []
        with torch.no_grad():
            for slat in slat_list:
                output_list.append(self.decode_slat(slat, formats))

            # if sparse_structure_optmizer_params['verbose']:
            for geo_slat in geo_slat_list:
                geo_output_list.append(self.decode_slat(geo_slat, ['gaussian']))

        ## geo_slat_list for visualzation
        
        
        if not return_optinit_noises:
            return output_list, geo_output_list, (z_s_list[-1], slat_list[-1])

        return output_list, geo_output_list, (z_s_list[-1], slat_list[-1]), {
            "sparse_init_noise": sparse_init_noise,
            "slat_init_noise": slat_init_noise,
        }
    
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
