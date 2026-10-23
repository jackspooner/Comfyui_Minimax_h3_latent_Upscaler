# Development instructions

## Worktree development

Run `COMFYUI_PYTHON=/path/to/prepared/comfyui/python python3 scripts/setup_worktree.py` before editing. The configured Python must be >=3.10 with PyTorch, einops and safetensors installed from the ComfyUI deployment's requirements. The check reuses that explicit runtime without copying dependencies, installing packages, loading models or connecting to a service. It is safe to rerun and leaves tracked files unchanged. GPU validation requires a compatible GPU/runtime and separately provisioned model files; unit tests use isolated fixtures.

Operating guidance belongs to the existing shared ComfyUI model-runtimes skill.

CPU verification: `PYTHONPATH=/path/to/ComfyUI OMP_NUM_THREADS=1 "$COMFYUI_PYTHON" -m pytest --confcutdir=tests -q tests`.
The ComfyUI source path must be the checkout under test, not an unrelated installed copy.
