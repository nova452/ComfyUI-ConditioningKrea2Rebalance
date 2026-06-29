import math

import torch

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _split_bands(t, n_bands=12):
    flat = t.shape[-1]
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    if ref_dir.shape[0] == 1 and target_batch != 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    if ref_dir.shape[0] != target_batch:
        ref_dir = ref_dir.mean(dim=0, keepdim=True).expand(target_batch, *ref_dir.shape[1:])
    return ref_dir


def _align_prompt(text):
    if text is None:
        return ""
    return str(text).replace("{", "{").replace("}", "}")


def _parse_floats(s):
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

# ignored
SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Target longest-side resolution per tier. Each image is scaled to it selected resolution independently
RESOLUTIONS = {"low": 256, "normal": 512, "high": 1024, "max": 1280}


def _scale_to_resolution(samples, target):

    n, c, h, w = samples.shape
    if h == target and w == target:
        return samples
    scale = target / max(h, w)
    nh = max(1, round(h * scale))
    nw = max(1, round(w * scale))
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def compile_edit(clip, prompt, images_with_size=None):
    """Encode an edit prompt with optional reference images."""
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

    images_vl = []
    image_prompt = ""

    if images_with_size:
        for i, (image, tier) in enumerate(images_with_size):
            if image is None:
                continue
            target = RESOLUTIONS.get(tier, 256)
            samples = image.movedim(-1, 1)  # NHWC -> NCHW
            scaled = _scale_to_resolution(samples, target)
            images_vl.append(scaled.movedim(1, -1))  # back to NHWC for clip.tokenize
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(len(images_vl))

    full_prompt = image_prompt + prompt if image_prompt else prompt

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl if images_vl else None,
        llama_template=SYS_TEMPLATE,
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    return conditioning


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=12):
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign=+1)
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(guidance_conditioning(item, ref_item, strength, per_band_strengths))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths)
        return structure
    return structure


def guidance(conditioning, reference, strength):
    return guidance_conditioning(conditioning, reference, strength, per_band_strengths=None)


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
        plw = _parse_floats(per_layer_weights) if per_layer_weights else None
        c = scale_conditioning(conditioning, multiplier, per_layer_weights=plw)
        return (c,)


class Krea2EditRebalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
        },
        "optional": {
            "negative": ("STRING", {"forceInput": True}),
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    @staticmethod
    def _process_cond(cond_main, cond_ref, refocus_strength=1.00, guidance_strength=1.000):
        cond_ref = refocus(
            cond_ref, refocus_strength, "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        cond_main = refocus(
            cond_main, refocus_strength, "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,12.0,0.0,0.0,0.0",
        )
        return guidance(cond_main, cond_ref, guidance_strength)

    def main(self, text, clip, refocus_strength=1.00, guidance_strength=1.000,
             negative=None,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        safe = _align_prompt(text)
        prompt_main = "" + safe
        ref_prefix = negative if negative is not None and str(negative) != "" else ""
        prompt_ref = str(ref_prefix) + ""

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        cond_raw = compile_edit(clip, prompt_main, None)

        if has_image:
            cond_image_main = compile_edit(clip, prompt_main, images_with_size)
            cond_image_ref = compile_edit(clip, prompt_ref, images_with_size)

            # compile the main cond with the subject layer before the 1st process (apply selective emphasis)
            cond_image_main = refocus(
                cond_image_main, 1.0,
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,12.0,0.0,0.0,0.0",
            )

            # unfocus the subject on ref
            cond_image_ref = refocus(
                cond_image_ref, 1.0,
                "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            )

            # 1st process: subject vs outlier guidance
            first = guidance(cond_image_main, cond_image_ref, guidance_strength)

            # post process: re-refocus the subject layer, multiplier 1
            compiled = refocus(
                first, 1.0,
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,12.0,0.0,0.0,0.0",
            )

            # 2: guider (unconditional vs compiled)
            second = guidance(cond_raw, compiled, -0.5)

            # 3: guider (first pipe vs second)
            third = guidance(first, second, -0.5)

            # step 4: custom Rebalance CFG with fixed schedules
            final = RebalanceCFG().main(
                cond_raw, third,
                "0.000-0.200:1.00;",
                "0.200-0.750:0.80; 0.750-0.875:1.40; 0.875-1.000:20.50",
                "gradual", 8,
            )[0]
        else:
            final = cond_raw

        return (final,)


class Krea2EncodeRebalance:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, text, clip,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "" + _align_prompt(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        final = compile_edit(clip, prompt, images_with_size if has_image else None)

        return (final,)


class RebalanceGuider:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "guidance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, positive, negative, guidance_strength=0.500):
        return (guidance(positive, negative, guidance_strength),)


class StepRebalance:
    """Split a conditioning schedule at a step threshold."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "step": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
            "bound": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, conditioning_2, step=0.00, bound=0.00):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Step Rebalance requires ComfyUI (node_helpers).")

        step = float(min(max(step, 0.0), 1.0))
        bound = float(min(max(bound, 0.0), 1.0))

        end_1 = float(min(step + bound, 1.0))
        start_2 = float(max(step - bound, 0.0))

        cond_split_1 = node_helpers.conditioning_set_values(
            conditioning_1, {"start_percent": 0.000, "end_percent": end_1},
        )
        cond_split_2 = node_helpers.conditioning_set_values(
            conditioning_2, {"start_percent": start_2, "end_percent": 1.000},
        )
        return (cond_split_2 + cond_split_1,)


def _parse_schedule(s):

    if not s:
        return None
    s = s.replace("\n", " ").replace("\t", " ").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    points = []
    for p in parts:
        if ":" in p:
            seg, mult_s = p.rsplit(":", 1)
        else:
            seg, mult_s = p, "1.0"
        if "-" in seg:
            start_s, end_s = seg.split("-", 1)
        else:
            start_s, end_s = seg, seg
        try:
            start = float(start_s.strip())
            end = float(end_s.strip())
            mult = float(mult_s.strip())
        except ValueError:
            continue
        start = float(min(max(start, 0.0), 1.0))
        end = float(min(max(end, 0.0), 1.0))
        if end < start:
            start, end = end, start
        points.append((start, end, mult))
    if not points:
        return None
    points.sort(key=lambda x: x[0])
    return points


def _build_schedule_segments(conditioning, points, interpolation, sub_steps=8):

    if not _COMFY_AVAILABLE:
        raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")
    out = []
    n = len(points)
    for i, (t0, t1, m_i) in enumerate(points):
        m_next = points[i + 1][2] if i + 1 < n else m_i
        if t1 <= t0:
            t1 = max(t1, t0)
            seg = scale_conditioning(conditioning, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)
            continue

        if interpolation == "gradual" and sub_steps > 1 and m_next != m_i:
            for k in range(sub_steps):
                f0 = k / sub_steps
                f1 = (k + 1) / sub_steps
                ts = t0 + (t1 - t0) * f0
                te = t0 + (t1 - t0) * f1
                # use the sub-segment midpoint for a stable linear ramp
                m = m_i + (m_next - m_i) * ((f0 + f1) / 2.0)
                seg = scale_conditioning(conditioning, m)
                seg = node_helpers.conditioning_set_values(
                    seg, {"start_percent": ts, "end_percent": te},
                )
                out.append(seg)
        else:
            seg = scale_conditioning(conditioning, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)

    combined = []
    for seg in out:
        combined = combined + seg
    return combined


class RebalanceCFG:
    """CFG-style conditioning rebalance from editable point schedule strings."""

    DEFAULT_SCHEDULE_1 = (
        "0.000-0.125:1.50; 0.125-0.250:1.40; 0.250-0.375:1.20; 0.375-0.500:1.00;"
        " 0.500-0.625:0.80; 0.625-0.750:0.60; 0.750-0.875:0.40; 0.875-1.000:0.20"
    )
    DEFAULT_SCHEDULE_2 = (
        "0.000-0.125:0.20; 0.125-0.250:0.40; 0.250-0.375:0.60; 0.375-0.500:0.80;"
        " 0.500-0.625:1.00; 0.625-0.750:1.20; 0.750-0.875:1.40; 0.875-1.000:1.50"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "schedule_1": ("STRING", {"default": cls.DEFAULT_SCHEDULE_1, "multiline": True}),
            "schedule_2": ("STRING", {"default": cls.DEFAULT_SCHEDULE_2, "multiline": True}),
            "interpolation": (["constant", "gradual"], {"default": "gradual"}),
            "sub_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, conditioning_2, schedule_1, schedule_2,
             interpolation="gradual", sub_steps=8):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")

        pts1 = _parse_schedule(schedule_1)
        pts2 = _parse_schedule(schedule_2)
        if pts1 is None or pts2 is None:
            raise ValueError(
                "Rebalance CFG: invalid schedule string. "
                "Use 'start-end:multiplier; ...' (e.g. 0.000-0.125:1.5; ...)."
            )

        sub_steps = int(min(max(sub_steps, 1), 64))
        seg1 = _build_schedule_segments(conditioning_1, pts1, interpolation, sub_steps)
        seg2 = _build_schedule_segments(conditioning_2, pts2, interpolation, sub_steps)
        return (seg1 + seg2,)


NODE_CLASS_MAPPINGS = {
    "Krea2EditRebalance": Krea2EditRebalance,
    "Krea2EncodeRebalance": Krea2EncodeRebalance,
    "RebalanceGuider": RebalanceGuider,
    "StepRebalance": StepRebalance,
    "RebalanceCFG": RebalanceCFG,
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
    "Krea2EncodeRebalance": "Krea 2 Encode Rebalance",
    "RebalanceGuider": "Rebalance Guider",
    "StepRebalance": "Step Rebalance",
    "RebalanceCFG": "Rebalance CFG Custom",
    "ConditioningKrea2Rebalance": "Conditioning Krea2 Rebalance",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
