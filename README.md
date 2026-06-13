# AnimaDualKVStyle

AnimaDualKVStyle is a custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implements a custom style Lora loader for the **Anima** model. It loads style adapters trained using the Dual KV Style Network and patches the attention blocks of the Anima diffusion model.

## Features

- **Anima Dual KV Style Lora Loader**: A dedicated custom node to load `.safetensors` style adapters.
- **Split-Half RoPE Support**: Implements exact native split-half Rotary Position Embedding matching the Anima model's RoPE implementation.
- **Chainable & Graph-Isolated Design**: Clones the model patcher configurations to allow loading multiple style adapters sequentially (chaining) without contamination across different branches of the workflow.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/TheDucker1/animadualkvstyle.git
   ```
2. Restart ComfyUI.

## Node Description: Anima Dual KV Style Lora Loader

### Inputs

- `model` (MODEL): The Anima diffusion model to patch.
- `lora_name` (select): The name of the style Lora file located in your `models/loras` folder.
- `strength` (FLOAT): Style multiplier strength (default `1.0`). Set to `0.0` to disable the style path.
- `apply_rope` (BOOLEAN): If enabled (default `True`), applies Rotary Position Embedding to the query states before style cross-attention.

### Outputs

- `MODEL`: The patched and style-conditioned Anima model.

## License

This project is licensed under the GNU General Public License v3.
