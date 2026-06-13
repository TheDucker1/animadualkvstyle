import os
import re
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
import folder_paths
import comfy.utils
import comfy.quant_ops

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class StylePatch:
    def __init__(self, k_style, v_style, k_norm_weight, down_weight, up_weight, strength, apply_rope):
        self.k_style = k_style
        self.v_style = v_style
        self.k_norm_weight = k_norm_weight
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.strength = strength
        self.apply_rope = apply_rope

    def apply(self, module, x, y, rope_emb):
        if self.strength == 0.0:
            return y

        # 1. Compute q from x exactly as in the original Attention block
        q = module.q_proj(x)
        # Rearrange to (B, S, H, D)
        q = q.view(q.shape[0], q.shape[1], module.n_heads, module.head_dim)
        q = module.q_norm(q)

        # 2. Apply RoPE if enabled (only for self-attention)
        if self.apply_rope and getattr(module, "is_selfattn", False) and rope_emb is not None:
            # Extract cos/sin from rope_emb of shape (B, S, 1, D_rope)
            cos = rope_emb[..., 0, 0].to(device=q.device, dtype=q.dtype)
            sin = rope_emb[..., 1, 0].to(device=q.device, dtype=q.dtype)
            
            d_rope = cos.shape[-1]
            q_rope_part = q[..., :d_rope]
            
            # Apply rotary positional embeddings using the split-half rotation formula
            q_rotated = (q_rope_part * cos) + (rotate_half(q_rope_part) * sin)
            
            # Concatenate back with the unrotated part of q
            q = torch.cat([q_rotated, q[..., d_rope:]], dim=-1)

        # 3. Get style keys and values, cast to correct device/dtype
        B = y.shape[0]
        k_style = self.k_style.to(device=x.device, dtype=x.dtype).repeat(B, 1, 1, 1)
        v_style = self.v_style.to(device=x.device, dtype=x.dtype).repeat(B, 1, 1, 1)

        # Apply RMSNorm to keys:
        variance = k_style.pow(2).mean(-1, keepdim=True)
        k_norm_weight = self.k_norm_weight.to(device=x.device, dtype=x.dtype)
        k_style_normed = k_style * torch.rsqrt(variance + 1e-6) * k_norm_weight

        # 4. Transpose to align heads: (B, H, S, D) and (B, H, N_queries, D)
        q_h = q.transpose(1, 2)
        k_h = k_style_normed.transpose(1, 2)
        v_h = v_style.transpose(1, 2)

        # Compute multi-head attention scores: (B, H, S, N_queries)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * (module.head_dim ** -0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute attention output: (B, H, S, D)
        out_h = torch.matmul(attn_weights, v_h)

        # Transpose and reshape back to sequence space: (B, S, inner_dim)
        out_style = out_h.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)

        # Project back to query_dim
        down_w = self.down_weight.to(device=x.device, dtype=x.dtype)
        up_w = self.up_weight.to(device=x.device, dtype=x.dtype)
        
        proj_down = F.linear(out_style, down_w)
        proj_up = F.linear(proj_down, up_w)
        
        out_style = proj_up * self.strength
        
        # Merge style attention path with main attention path
        return y + out_style

def patch_attention_module(module):
    # Only patch if not already patched
    if not hasattr(module, "_style_patched"):
        module._style_patched = True
        module._original_forward = module.forward
        
        def custom_forward(self, x, *args, **kwargs):
            # Run the original forward pass
            y = self._original_forward(x, *args, **kwargs)
            
            # Extract transformer_options
            transformer_options = kwargs.get("transformer_options", None)
            if transformer_options is None and len(args) >= 3:
                transformer_options = args[2]
                
            if transformer_options is None:
                return y
                
            # Retrieve the style patches mapped to this module's unique ID
            style_patches = transformer_options.get("style_patches", None)
            if not style_patches:
                return y
                
            patches = style_patches.get(id(self), None)
            if not patches:
                return y
                
            # Extract rope_emb (index 1 in args: context is args[0], rope_emb is args[1])
            rope_emb = kwargs.get("rope_emb", None)
            if rope_emb is None and len(args) >= 2:
                rope_emb = args[1]
                
            # Apply each style patch sequentially
            for patch in patches:
                y = patch.apply(self, x, y, rope_emb)
                
            return y
            
        module.forward = types.MethodType(custom_forward, module)

class AnimaLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to apply the style to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The style adapter Lora model to load."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01, "tooltip": "Style strength multiplier."}),
                "apply_rope": ("BOOLEAN", {"default": True, "tooltip": "Apply Rotary Position Embedding to Q before cross attend with custom KV path (experimental)."})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_style_lora"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads a Dual KV Style Lora and patches the attention blocks of the Anima diffusion model."

    def load_style_lora(self, model, lora_name, strength, apply_rope):
        if strength <= 0.0:
            return (model,)

        # Clone the model patcher to preserve graph isolation
        model_patched = model.clone()

        # Load safetensors file
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = safetensors.torch.load_file(lora_path)

        # Walk named modules of the diffusion model to find target attention modules
        attention_modules = {}
        for name, module in model_patched.model.diffusion_model.named_modules():
            if "llm_adapter" in name:
                continue
            cls = module.__class__.__name__
            if cls == "Attention":
                # Format block names to match key prefixes (e.g. style_kv_dit_blocks_5_self_attn)
                prefix = f"style_kv_dit_{name}".replace(".", "_")
                attention_modules[prefix] = module

        # Prepare/extract style weights and hook the target modules
        to = model_patched.model_options.setdefault("transformer_options", {})
        style_patches = to.setdefault("style_patches", {})

        patched_any = False
        for prefix, module in attention_modules.items():
            k_style_key = f"{prefix}.k_style"
            if k_style_key in sd:
                # Extract parameters for this prefix
                k_style = sd[k_style_key]
                v_style = sd[f"{prefix}.v_style"]
                k_norm_weight = sd[f"{prefix}.k_norm_style.weight"]
                down_weight = sd[f"{prefix}.out_proj_down.weight"]
                up_weight = sd[f"{prefix}.out_proj_up.weight"]

                # Patch the Attention module's forward function instance
                patch_attention_module(module)

                # Register the style patch under the module's unique ID
                if id(module) not in style_patches:
                    style_patches[id(module)] = []
                style_patches[id(module)].append(
                    StylePatch(k_style, v_style, k_norm_weight, down_weight, up_weight, strength, apply_rope)
                )
                patched_any = True

        if not patched_any:
            print(f"Warning: No matching Dual KV layers found in {lora_name} for the active model.")

        return (model_patched,)

NODE_CLASS_MAPPINGS = {
    "AnimaLoraLoader": AnimaLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoraLoader": "Anima Dual KV Style Lora Loader"
}
