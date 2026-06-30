"""Ideogram 4 (Qwen3-VL-8B, 13-layer tap) specific conditioning rebalance nodes."""

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


# 13-layer tap of Qwen3-VL-8B (comfy captures layer inputs, offset by +1).
IDEOGRAM4_TAP_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36]
IDEOGRAM4_N_TAPS = len(IDEOGRAM4_TAP_LAYERS)            # 13
IDEOGRAM4_HIDDEN_DIM = 4096
IDEOGRAM4_FEATURE_DIM = IDEOGRAM4_N_TAPS * IDEOGRAM4_HIDDEN_DIM   # 53248

# Register the Ideogram 4 profile with the core detection system.
core.register_encoder_profile(
    "ideogram4",
    n_taps=IDEOGRAM4_N_TAPS,
    hidden_dim=IDEOGRAM4_HIDDEN_DIM,
    tap_layers=IDEOGRAM4_TAP_LAYERS,
)

# Ignored
IDEOGRAM4_SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit_ideogram4(clip, prompt, images_with_size=None):

    return compile_edit(clip, prompt, images_with_size, llama_template=None)


class ConditioningIdeogram4Rebalance:

    # 13 weights
    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"

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


class Ideogram4EditRebalance:
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
        # 13-band refocus: keep ref uniform, isolate the subject band (index 8).
        cond_ref = refocus(
            cond_ref, refocus_strength,
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        cond_main = refocus(
            cond_main, refocus_strength,
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        return guidance(cond_main, cond_ref, guidance_strength)

    def main(self, text, clip, refocus_strength=1.00, guidance_strength=1.000,
             negative=None,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Ideogram 4 Edit requires ComfyUI (comfy.utils, node_helpers).")

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

        cond_raw = compile_edit_ideogram4(clip, prompt_main, None)

        if has_image:
            cond_image_main = compile_edit_ideogram4(clip, prompt_main, images_with_size)
            cond_image_ref = compile_edit_ideogram4(clip, prompt_ref, images_with_size)

            # compile the main cond with the subject layer before the 1st process (apply selective emphasis)
            cond_image_main = refocus(
                cond_image_main, 1.0,
                "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            )

            # unfocus the subject on ref
            cond_image_ref = refocus(
                cond_image_ref, 1.0,
                "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            )

            # 1st process: subject vs outlier guidance
            first = guidance(cond_image_main, cond_image_ref, guidance_strength)

            # post process: re-refocus the subject layer, multiplier 1
            compiled = refocus(
                first, 1.0,
                "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
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


class Ideogram4EncodeRebalance:

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
            raise RuntimeError("Ideogram 4 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "" + _align_prompt(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        final = compile_edit_ideogram4(clip, prompt, images_with_size if has_image else None)

        return (final,)


NODE_CLASS_MAPPINGS = {
    "ConditioningIdeogram4Rebalance": ConditioningIdeogram4Rebalance,
    "Ideogram4EditRebalance": Ideogram4EditRebalance,
    "Ideogram4EncodeRebalance": Ideogram4EncodeRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningIdeogram4Rebalance": "Conditioning Ideogram4 Rebalance",
    "Ideogram4EditRebalance": "Ideogram 4 Image Edit Rebalance",
    "Ideogram4EncodeRebalance": "Ideogram 4 Encode Rebalance",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ConditioningIdeogram4Rebalance",
    "Ideogram4EditRebalance",
    "Ideogram4EncodeRebalance",
    "IDEOGRAM4_TAP_LAYERS",
    "IDEOGRAM4_FEATURE_DIM",
]
