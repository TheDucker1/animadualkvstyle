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

class StyleLoraPatchManager:
    def __init__(self, patches_list):
        self.patches_list = patches_list
        self.org_forwards = {}

    def apply_to(self):
        # Group patches by module in case multiple style LoRAs patch the same module
        grouped = {}
        for module, patch in self.patches_list:
            grouped.setdefault(module, []).append(patch)

        for module, plist in grouped.items():
            if module not in self.org_forwards:
                self.org_forwards[module] = module.forward

                def make_custom_forward(mod, org_f, plist_inner):
                    def custom_forward(self_mod, x, *args, **kwargs):
                        # Run the original forward pass
                        y = org_f(x, *args, **kwargs)

                        # Extract transformer_options
                        transformer_options = kwargs.get("transformer_options", None)
                        if transformer_options is None and len(args) >= 3:
                            transformer_options = args[2]

                        if transformer_options is None:
                            return y

                        # Extract rope_emb (index 1 in args: context is args[0], rope_emb is args[1])
                        rope_emb = kwargs.get("rope_emb", None)
                        if rope_emb is None and len(args) >= 2:
                            rope_emb = args[1]

                        # Determine positive/negative chunks based on cond_or_uncond
                        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
                        pos_indices = None
                        neg_index = None
                        if isinstance(cond_or_uncond, list):
                            pos_indices = [i for i, val in enumerate(cond_or_uncond) if val == 0]
                            if 1 in cond_or_uncond:
                                neg_index = cond_or_uncond.index(1)

                        # Subset input x to only positive conditioning chunks to avoid useless negative style computation
                        x_pos = x
                        if pos_indices is not None and len(pos_indices) < x.shape[0]:
                            try:
                                x_chunks = x.chunk(x.shape[0], dim=0)
                                x_pos = torch.cat([x_chunks[i] for i in pos_indices], dim=0)
                            except Exception:
                                pos_indices = None

                        # Compute cumulative style attention output only on the positive chunks
                        dummy_y = torch.zeros(x_pos.shape[0], y.shape[1], y.shape[2], dtype=y.dtype, device=y.device)
                        out_style_pos = dummy_y
                        for patch in plist_inner:
                            out_style_pos = patch.apply(self_mod, x_pos, out_style_pos, rope_emb)

                        # --- BEGIN Anima-NAG node compatibility branch ---
                        # This branch resolves compatibility with the ComfyUI-Anima-Nag node.
                        # Since both nodes patch the model configuration concurrently, we inspect the NAG node's
                        # optimized_attention_override function closure to dynamically retrieve the guidance parameters
                        # (scale, tau, alpha, and active sigma range) without modifying the external NAG repository.
                        nag_scale = nag_tau = nag_alpha = None
                        nag_override = transformer_options.get("optimized_attention_override", None)
                        if nag_override is not None and nag_override.__class__.__name__ == 'function':
                            try:
                                # Dynamically inspect NAG's local variables captured within its attention_override closure
                                free_vars = nag_override.__code__.co_freevars
                                closure_dict = {
                                    name: cell.cell_contents
                                    for name, cell in zip(free_vars, nag_override.__closure__)
                                }
                                nag_scale = closure_dict.get("scale", None)
                                nag_tau = closure_dict.get("tau", None)
                                nag_alpha = closure_dict.get("alpha", None)
                                sigma_start = closure_dict.get("sigma_start", None)
                                sigma_end = closure_dict.get("sigma_end", None)
                            except Exception:
                                pass

                        # Determine if Normalized Attention Guidance (NAG) is active for the current sampling step.
                        # NAG must have positive/negative chunks in the batch and the current step's sigma must fall 
                        # within the configured active sigma range.
                        is_nag_active_for_step = False
                        if (
                            nag_scale is not None and nag_scale > 0.0 and
                            nag_tau is not None and nag_alpha is not None and
                            pos_indices is not None and neg_index is not None
                        ):
                            sigmas = transformer_options.get("sigmas", None)
                            if sigmas is not None and len(sigmas) > 0:
                                try:
                                    sigma_val = float(sigmas[0])
                                    if sigma_end < sigma_val <= sigma_start:
                                        is_nag_active_for_step = True
                                except Exception:
                                    pass

                        if is_nag_active_for_step:
                            try:
                                # Under NAG, we apply guidance and normalization parameters to the positive chunks of the style LoRA.
                                # This ensures the style features are scaled by the NAG normalization ratio so they do not
                                # overpower the guided prompt or cause prompt distortion.
                                y_chunks = y.chunk(len(cond_or_uncond), dim=0)
                                out_style_pos_chunks = out_style_pos.chunk(len(pos_indices), dim=0)

                                y_neg_base = torch.cat([y_chunks[neg_index]] * len(pos_indices), dim=0)
                                out_chunks = list(y_chunks)

                                for i, pos_index in enumerate(pos_indices):
                                    y_pos_base = y_chunks[pos_index]
                                    style_pos = out_style_pos_chunks[i]

                                    # 1. Compute the guided positive branch output (base attention + guided prompt subtraction)
                                    y_tilde_base = y_pos_base + nag_scale * (y_pos_base - y_neg_base)

                                    # 2. Compute norms of the combined positive branch (base + style) to evaluate the scaling ratio
                                    eps = 1e-6
                                    norm_pos = torch.norm(y_pos_base + style_pos, p=1, dim=-1, keepdim=True).clamp_min(eps)
                                    norm_tilde = torch.norm(y_tilde_base + style_pos, p=1, dim=-1, keepdim=True).clamp_min(eps)

                                    # 3. Calculate NAG's normalization ratio and final blend scaling factor
                                    ratio = norm_tilde / norm_pos
                                    scaling_factor = nag_alpha * (torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio) + (1.0 - nag_alpha)

                                    # 4. Scale the positive style output chunk and combine it
                                    out_chunks[pos_index] = y_pos_base + scaling_factor * style_pos

                                # The negative chunk remains completely clean (Option A: no style added to the unconditioned branch)
                                out_chunks[neg_index] = y_chunks[neg_index]

                                return torch.cat(out_chunks, dim=0)
                            except Exception as e:
                                # Fallback to standard addition in case of unexpected execution failures
                                print(f"[AnimaDualKVStyle] NAG compatibility error, falling back: {e}")
                                return y + out_style_pos
                        # --- END Anima-NAG node compatibility branch ---

                        # Default branch: positive-only style addition
                        try:
                            if pos_indices is not None:
                                y_chunks = y.chunk(y.shape[0], dim=0)
                                out_style_pos_chunks = out_style_pos.chunk(len(pos_indices), dim=0)

                                out_chunks = list(y_chunks)
                                for i, pos_index in enumerate(pos_indices):
                                    out_chunks[pos_index] = y_chunks[pos_index] + out_style_pos_chunks[i]

                                # Negative chunk remains completely clean
                                if neg_index is not None:
                                    out_chunks[neg_index] = y_chunks[neg_index]

                                return torch.cat(out_chunks, dim=0)
                        except Exception:
                            pass

                        return y + out_style_pos
                    return custom_forward

                module.forward = types.MethodType(make_custom_forward(module, module.forward, plist), module)

    def restore(self):
        for module, org_forward in self.org_forwards.items():
            module.forward = org_forward
        self.org_forwards.clear()

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

        patches_list = []
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

                patch = StylePatch(k_style, v_style, k_norm_weight, down_weight, up_weight, strength, apply_rope)
                patches_list.append((module, patch))
                patched_any = True

        if not patched_any:
            print(f"Warning: No matching Dual KV layers found in {lora_name} for the active model.")
            return (model_patched,)

        manager = StyleLoraPatchManager(patches_list)
        old_wrapper = model_patched.model_options.get("model_function_wrapper")

        def _call_next(apply_model, input_x, timestep, c):
            if old_wrapper is not None:
                return old_wrapper(apply_model, {"input": input_x, "timestep": timestep, "c": c})
            return apply_model(input_x, timestep, **c)

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Put style patches into transformer_options.style_patches so they are visible
            # to any other nodes or hooks checking it during execution.
            to = c.setdefault("transformer_options", {})
            style_patches = to.setdefault("style_patches", {})

            for module, patch in patches_list:
                style_patches.setdefault(id(module), []).append(patch)

            # Apply the attention forwards patch dynamically
            manager.apply_to()
            try:
                return _call_next(apply_model, input_x, timestep, c)
            finally:
                manager.restore()
                # Clean up style_patches from this run
                for module, patch in patches_list:
                    if id(module) in style_patches:
                        try:
                            style_patches[id(module)].remove(patch)
                            if not style_patches[id(module)]:
                                del style_patches[id(module)]
                        except ValueError:
                            pass

        model_patched.set_model_unet_function_wrapper(wrapper)
        return (model_patched,)

NODE_CLASS_MAPPINGS = {
    "AnimaLoraLoader": AnimaLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoraLoader": "Anima Dual KV Style Lora Loader"
}

