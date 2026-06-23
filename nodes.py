import torch


def _scale_cond_tensor(t: torch.Tensor, multiplier, per_layer_weights=None):
    """Scale a conditioning tensor, optionally with per-layer weighting.

    Krea2 conditioning arrives as (B, seq, 12*2560) — the 12 Qwen3-VL taps flattened
    into the feature dim.  When per_layer_weights is given we reshape to
    (B, seq, 12, D), apply a different gain to each tap and flatten back.
    """
    if per_layer_weights is None:
        return t * multiplier

    flat = t.shape[-1]
    n_layers = len(per_layer_weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(per_layer_weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * multiplier
    return t * multiplier


def _parse_per_layer(s: str):
    """Parse a comma-separated list of floats. Returns None if empty/invalid."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals


def scale_conditioning(structure, multiplier, per_layer_weights=None):
    """leaving masks / pooled output intact."""
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, multiplier, per_layer_weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, multiplier, per_layer_weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, multiplier, per_layer_weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, multiplier, per_layer_weights)
                for k, v in structure.items()}
    return structure


class ConditioningKrea2Rebalance:

    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "multiplier": ("FLOAT", {"default": 4.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01}),
            "per_layer_weights": ("STRING", {"default": cls.DEFAULT_WEIGHTS, "multiline": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning, multiplier, per_layer_weights=None):
        plw = _parse_per_layer(per_layer_weights) if per_layer_weights else None
        c = scale_conditioning(conditioning, multiplier, per_layer_weights=plw)
        return (c,)
