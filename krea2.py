"""Krea 2 (Qwen3-VL-4B, 12-layer tap) specific conditioning rebalance nodes."""

import torch

from . import conditioning_rebalance as core
from .conditioning_rebalance import (
    compile_edit,
    guidance,
    refocus,
    _align_prompt,
)

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


# 12-layer tap of Qwen3-VL-4B (tap k == hidden_states[k], no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
KREA2_N_TAPS = len(KREA2_TAP_LAYERS)          # 12
KREA2_HIDDEN_DIM = 2560
KREA2_FEATURE_DIM = KREA2_N_TAPS * KREA2_HIDDEN_DIM   # 30720

# Register the Krea 2 profile with the core detection system.
core.register_encoder_profile(
    "krea2",
    n_taps=KREA2_N_TAPS,
    hidden_dim=KREA2_HIDDEN_DIM,
    tap_layers=KREA2_TAP_LAYERS,
)

# System template used by Krea 2 image-edit conditioning.
KREA2_SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit_krea2(clip, prompt, images_with_size=None):
    """Encode a Krea 2 edit prompt with optional reference images."""
    return compile_edit(clip, prompt, images_with_size, llama_template=KREA2_SYS_TEMPLATE)


class ConditioningKrea2Rebalance:
    """Per-layer conditioning scaler for Krea 2's layout."""

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
        plw = core._parse_floats(per_layer_weights) if per_layer_weights else None
        c = core.scale_conditioning(conditioning, multiplier, weights=plw)
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

        cond_raw = compile_edit_krea2(clip, prompt_main, None)

        if has_image:
            cond_image_main = compile_edit_krea2(clip, prompt_main, images_with_size)
            cond_image_ref = compile_edit_krea2(clip, prompt_ref, images_with_size)

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
            final = core.RebalanceCFG().main(
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

        final = compile_edit_krea2(clip, prompt, images_with_size if has_image else None)

        return (final,)


NODE_CLASS_MAPPINGS = {
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,
    "Krea2EditRebalance": Krea2EditRebalance,
    "Krea2EncodeRebalance": Krea2EncodeRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningKrea2Rebalance": "Conditioning Krea2 Rebalance",
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
    "Krea2EncodeRebalance": "Krea 2 Encode Rebalance",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ConditioningKrea2Rebalance",
    "Krea2EditRebalance",
    "Krea2EncodeRebalance",
    "KREA2_TAP_LAYERS",
    "KREA2_FEATURE_DIM",
]
