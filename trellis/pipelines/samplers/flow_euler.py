from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _randn_like(self, x):
        if torch.is_tensor(x):
            return torch.randn_like(x)
        if hasattr(x, 'replace') and hasattr(x, 'feats'):
            return x.replace(torch.randn_like(x.feats))
        raise ValueError(f"Unsupported type for random generation: {type(x)}")

    def _blend_repaint_region(self, repaint_sample, ref_sample, repaint_mask):
        """
        Blend two samples with repaint mask.
        True in repaint_mask means repaint (use repaint_sample),
        False means preserve (use ref_sample).
        """
        if torch.is_tensor(repaint_sample):
            if not torch.is_tensor(repaint_mask):
                return repaint_sample
            mask = repaint_mask.to(device=repaint_sample.device)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            if mask.shape[0] == 1 and repaint_sample.shape[0] > 1:
                mask = mask.expand(repaint_sample.shape[0], *mask.shape[1:])
            if mask.ndim == repaint_sample.ndim - 1:
                mask = mask.unsqueeze(1)
            if mask.ndim != repaint_sample.ndim:
                return repaint_sample
            if mask.shape[1] == 1 and repaint_sample.shape[1] > 1:
                mask = mask.expand(-1, repaint_sample.shape[1], *mask.shape[2:])
            return torch.where(mask, repaint_sample, ref_sample)

        if hasattr(repaint_sample, 'replace') and hasattr(repaint_sample, 'feats'):
            if not torch.is_tensor(repaint_mask):
                return repaint_sample
            mask = repaint_mask.to(device=repaint_sample.device)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            if mask.ndim == 1:
                mask = mask.unsqueeze(1)
            if mask.ndim != 2:
                return repaint_sample
            if mask.shape[0] != repaint_sample.feats.shape[0]:
                return repaint_sample
            if mask.shape[1] == 1 and repaint_sample.feats.shape[1] > 1:
                mask = mask.expand(-1, repaint_sample.feats.shape[1])
            if mask.shape[1] != repaint_sample.feats.shape[1]:
                return repaint_sample
            feats = torch.where(mask, repaint_sample.feats, ref_sample.feats)
            return repaint_sample.replace(feats)

        return repaint_sample

    def _to_float_mask_like(self, x_like, mask):
        if not torch.is_tensor(mask):
            return None
        if torch.is_tensor(x_like):
            m = mask.to(device=x_like.device, dtype=x_like.dtype)
            if m.shape[0] == 1 and x_like.shape[0] > 1:
                m = m.expand(x_like.shape[0], *m.shape[1:])
            if m.ndim == x_like.ndim - 1:
                m = m.unsqueeze(1)
            if m.ndim != x_like.ndim:
                return None
            if m.shape[1] == 1 and x_like.shape[1] > 1:
                m = m.expand(-1, x_like.shape[1], *m.shape[2:])
            return m.clamp(0, 1)

        if hasattr(x_like, 'feats'):
            m = mask.to(device=x_like.device, dtype=x_like.feats.dtype)
            if m.ndim == 1:
                m = m.unsqueeze(1)
            if m.ndim != 2:
                return None
            if m.shape[0] != x_like.feats.shape[0]:
                return None
            if m.shape[1] == 1 and x_like.feats.shape[1] > 1:
                m = m.expand(-1, x_like.feats.shape[1])
            if m.shape[1] != x_like.feats.shape[1]:
                return None
            return m.clamp(0, 1)
        return None

    def _masked_lerp(self, generated_sample, ref_sample, edit_mask, edit_strength: float = 1.0):
        """
        Soft blend:
        - edit_mask == 1 means keep generated sample
        - edit_mask == 0 means keep reference sample
        """
        if edit_strength < 1.0 and torch.is_tensor(edit_mask):
            mask = edit_mask.to(dtype=torch.float32) * float(edit_strength)
        else:
            mask = edit_mask
        m = self._to_float_mask_like(generated_sample, mask)
        if m is None:
            return self._blend_repaint_region(generated_sample, ref_sample, edit_mask)

        if torch.is_tensor(generated_sample):
            return m * generated_sample + (1.0 - m) * ref_sample
        if hasattr(generated_sample, 'replace') and hasattr(generated_sample, 'feats'):
            feats = m * generated_sample.feats + (1.0 - m) * ref_sample.feats
            return generated_sample.replace(feats)
        return generated_sample

    def _lowpass_like(self, x, downsample_factor: int = 4):
        if downsample_factor <= 1:
            return x
        if not torch.is_tensor(x):
            raise ValueError("ILVR low-pass is only supported for dense tensors.")
        if x.ndim != 5:
            raise ValueError(f"Expected rank-5 tensor for ILVR low-pass, got rank-{x.ndim}.")
        k = int(downsample_factor)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        return F.avg_pool3d(x, kernel_size=k, stride=1, padding=pad)

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def repaint(
        self,
        model,
        noise,
        x0,
        repaint_mask,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Repaint samples by anchoring non-repaint region to the forward-noised base x0.
        """
        sample = noise
        eps_ref = self._randn_like(x0)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Repainting", disable=not verbose):
            ref_t = (1 - t) * x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
            sample = self._blend_repaint_region(sample, ref_t, repaint_mask)
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        sample = self._blend_repaint_region(sample, x0, repaint_mask)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sdedit(
        self,
        model,
        x0,
        cond: Optional[Any] = None,
        edit_mask=None,
        ref_x0=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        strength: float = 0.6,
        verbose: bool = True,
        **kwargs
    ):
        """
        SDEdit by starting from a forward-noised base x0 at t_start=strength and denoising to 0.
        """
        if ref_x0 is None:
            ref_x0 = x0
        t_start = float(np.clip(strength, 0.0, 1.0))
        if t_start <= 0.0:
            ret = edict({"samples": x0, "pred_x_t": [], "pred_x_0": []})
            return ret

        eps = self._randn_like(x0)
        eps_ref = self._randn_like(ref_x0)
        sample = (1 - t_start) * x0 + (self.sigma_min + (1 - self.sigma_min) * t_start) * eps
        t_seq = np.linspace(t_start, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="SDEdit", disable=not verbose):
            if edit_mask is not None:
                ref_t = (1 - t) * ref_x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
                sample = self._blend_repaint_region(sample, ref_t, edit_mask)
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        if edit_mask is not None:
            sample = self._blend_repaint_region(sample, ref_x0, edit_mask)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def blended_diffusion(
        self,
        model,
        noise,
        x0,
        blend_mask,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        blend_strength: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Blended diffusion baseline.
        Blends sample with forward-noised reference outside edit region each step.
        """
        sample = noise
        eps_ref = self._randn_like(x0)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="BlendedDiff", disable=not verbose):
            ref_t = (1 - t) * x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
            sample = self._masked_lerp(sample, ref_t, blend_mask, edit_strength=blend_strength)
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        sample = self._masked_lerp(sample, x0, blend_mask, edit_strength=blend_strength)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def ilvr(
        self,
        model,
        x0,
        cond: Optional[Any] = None,
        edit_mask=None,
        ref_x0=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        strength: float = 0.6,
        downsample_factor: int = 4,
        ilvr_weight: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        ILVR baseline (dense tensor only): apply low-frequency constraint in preserve region.
        """
        if not torch.is_tensor(x0):
            raise NotImplementedError("ILVR sampler currently supports dense tensors only.")
        if ref_x0 is None:
            ref_x0 = x0
        t_start = float(np.clip(strength, 0.0, 1.0))
        if t_start <= 0.0:
            return edict({"samples": x0, "pred_x_t": [], "pred_x_0": []})

        eps = self._randn_like(x0)
        eps_ref = self._randn_like(ref_x0)
        sample = (1 - t_start) * x0 + (self.sigma_min + (1 - self.sigma_min) * t_start) * eps
        t_seq = np.linspace(t_start, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        mask = self._to_float_mask_like(sample, edit_mask)
        preserve = (1.0 - mask) if mask is not None else None

        for t, t_prev in tqdm(t_pairs, desc="ILVR", disable=not verbose):
            ref_t = (1 - t) * ref_x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
            ref_t_prev = (1 - t_prev) * ref_x0 + (self.sigma_min + (1 - self.sigma_min) * t_prev) * eps_ref
            if edit_mask is not None:
                sample = self._blend_repaint_region(sample, ref_t, edit_mask)

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev

            lf_sample = self._lowpass_like(sample, downsample_factor=downsample_factor)
            lf_ref = self._lowpass_like(ref_t_prev, downsample_factor=downsample_factor)
            if preserve is None:
                sample = sample + ilvr_weight * (lf_ref - lf_sample)
            else:
                sample = sample + ilvr_weight * preserve * (lf_ref - lf_sample)

            if edit_mask is not None:
                sample = self._blend_repaint_region(sample, ref_t_prev, edit_mask)
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)

        if edit_mask is not None:
            sample = self._blend_repaint_region(sample, ref_x0, edit_mask)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def dps(
        self,
        model,
        noise,
        x0,
        dps_mask,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        dps_weight: float = 0.3,
        verbose: bool = True,
        **kwargs
    ):
        """
        Diffusion Posterior Sampling baseline with masked data-consistency correction.
        """
        sample = noise
        eps_ref = self._randn_like(x0)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        mask = self._to_float_mask_like(sample, dps_mask)
        preserve = (1.0 - mask) if mask is not None else None

        for t, t_prev in tqdm(t_pairs, desc="DPS", disable=not verbose):
            ref_t = (1 - t) * x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev

            if torch.is_tensor(sample):
                if preserve is None:
                    sample = sample + dps_weight * (ref_t - sample)
                else:
                    sample = sample + dps_weight * preserve * (ref_t - sample)
            elif hasattr(sample, 'replace') and hasattr(sample, 'feats'):
                if preserve is None:
                    feats = sample.feats + dps_weight * (ref_t.feats - sample.feats)
                else:
                    feats = sample.feats + dps_weight * preserve * (ref_t.feats - sample.feats)
                sample = sample.replace(feats)

            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)

        sample = self._blend_repaint_region(sample, x0, dps_mask)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def multidiffusion(
        self,
        model,
        noise,
        x0,
        mask_list,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        weights: Optional[List[float]] = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        MultiDiffusion baseline:
        run multiple masked denoising branches and aggregate predictions on overlapping masks.
        """
        if mask_list is None or len(mask_list) == 0:
            raise ValueError("mask_list must be non-empty for multidiffusion.")
        if weights is None:
            weights = [1.0] * len(mask_list)
        if len(weights) != len(mask_list):
            raise ValueError("weights length must match mask_list length.")

        sample = noise
        eps_ref = self._randn_like(x0)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="MultiDiff", disable=not verbose):
            ref_t = (1 - t) * x0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps_ref
            branch_preds = []
            branch_weights = []
            branch_pred_x0 = []
            for w, mask in zip(weights, mask_list):
                local_sample = self._blend_repaint_region(sample, ref_t, mask)
                out = self.sample_once(model, local_sample, t, t_prev, cond, **kwargs)
                branch_preds.append(out.pred_x_prev)
                branch_pred_x0.append(out.pred_x_0)
                branch_weights.append(float(w))

            if torch.is_tensor(sample):
                agg = torch.zeros_like(sample)
                wsum = torch.zeros_like(sample)
                for pred_i, m_i, w_i in zip(branch_preds, mask_list, branch_weights):
                    m = self._to_float_mask_like(sample, m_i)
                    if m is None:
                        continue
                    agg = agg + w_i * m * pred_i
                    wsum = wsum + w_i * m
                sample = torch.where(wsum > 1e-6, agg / (wsum + 1e-6), ref_t)
            elif hasattr(sample, 'replace') and hasattr(sample, 'feats'):
                agg = torch.zeros_like(sample.feats)
                wsum = torch.zeros_like(sample.feats)
                for pred_i, m_i, w_i in zip(branch_preds, mask_list, branch_weights):
                    m = self._to_float_mask_like(sample, m_i)
                    if m is None:
                        continue
                    agg = agg + w_i * m * pred_i.feats
                    wsum = wsum + w_i * m
                feats = torch.where(wsum > 1e-6, agg / (wsum + 1e-6), ref_t.feats)
                sample = sample.replace(feats)
            else:
                sample = branch_preds[-1]

            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(branch_pred_x0[-1])

        sample = self._blend_repaint_region(sample, x0, mask_list[0])
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
