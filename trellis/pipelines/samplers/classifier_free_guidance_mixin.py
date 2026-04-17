from typing import *
import torch


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, optinit_mask=None, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        if optinit_mask is None:
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred

        # Keep cfg strength in masked-True region and use 2x cfg in masked-False region.
        mask = optinit_mask
        if not torch.is_tensor(mask):
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        
        mask = mask.to(device=pred.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5

        if mask.shape[0] == 1 and pred.shape[0] > 1:
            mask = mask.expand(pred.shape[0], *mask.shape[1:])
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(1)
        if mask.ndim != pred.ndim:
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand(-1, pred.shape[1], *mask.shape[2:])

        cfg_tensor = torch.as_tensor(cfg_strength, device=pred.device, dtype=pred.dtype)
        cfg_map = torch.where(mask, cfg_tensor, cfg_tensor * 3.0)
        return pred + cfg_map * (pred - neg_pred)
