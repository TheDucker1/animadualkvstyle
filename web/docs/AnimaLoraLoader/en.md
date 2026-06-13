# Anima Dual KV Style Lora Loader

Loads a custom `.safetensors` style adapter trained using the Dual KV Style Network, and dynamically patches the `Attention` modules of the Anima diffusion model.

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | MODEL | The Anima diffusion model to apply the style to. |
| `lora_name` | select | The style adapter Lora model to load from the `loras/` directory. |
| `strength` | FLOAT | The style strength multiplier. Can be positive or negative (default `1.0`). |
| `apply_rope` | BOOLEAN | If True, applies Rotary Position Embedding to the query vector before cross-attention with the style KV path. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | The patched, style-conditioned diffusion model. |
