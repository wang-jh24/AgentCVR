# CrossVid Agent Quick Start

Use this repository to run CrossVid video-understanding Agent GRPO training. Two launcher scripts are provided for **4B** and **8B** models; choose one based on GPU memory and needs.

## Environment setup

### 1. Create and activate conda environment

```bash
conda create -n verl python=3.12
conda activate verl
```

### 2. Install dependencies (vLLM, SGLang, Megatron-Core, etc.)

From the **repository root**:

```bash
bash scripts/install_vllm_sglang_mcore.sh
```

This script installs CUDA-related dependencies, PyTorch, vLLM, SGLang, Megatron-Core, and other packages required for training. Ensure you have a supported CUDA environment before running it.

### 3. Optional

- Set `WANDB_API_KEY` in your environment if you want to log to Weights & Biases.

## Data preparation

1. Preprocess CrossVid data to parquet format (see `examples/data_preprocess/README_CROSSVID.md` and `README_CROSSVID_MINI.md`).
2. Default training data path: `examples/data_preprocess/format_data/crossvid_train_mini.parquet`.  
   To use a different path, set the environment variable `CROSSVID_DATA_FILE`.

## Run training

From the **repository root**, run one of the following:

### 4B model (lower GPU memory)

```bash
chmod +x crossvid_agent_4b.sh
./crossvid_agent_4b.sh
```

- Default actor: `Qwen/Qwen3-4B`
- Default learning rate: `1e-7`, total epochs: 2
- Default wandb experiment name: `agent_4b`

### 8B model

```bash
chmod +x crossvid_agent_8b.sh
./crossvid_agent_8b.sh
```

- Default actor: `Qwen/Qwen3-8B`
- Default learning rate: `5e-7`, total epochs: 3
- Default wandb experiment name: `agent`

Both scripts use the same config file `examples/sglang_multiturn/config/crossvid_qwen3_grpo.yaml`; only model path, learning rate, epochs, etc. are overridden via the launcher.

## Optional environment variables

Both scripts support the following environment variables (defaults below; 4B script differs only in some values):

| Variable | Description | 4B default | 8B default |
|----------|-------------|------------|------------|
| `CROSSVID_DATA_FILE` | Path to train/val parquet | `.../crossvid_train_mini.parquet` | same |
| `CROSSVID_MODEL_PATH` | Actor model path (HuggingFace or local) | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` |
| `CROSSVID_CRITIC_PATH` | Critic model path | `Qwen/Qwen-Tiny-Critic` | same |
| `CROSSVID_CONFIG_DIR` | Config directory | `examples/sglang_multiturn/config` | same |
| `CROSSVID_PROJECT_NAME` | wandb project name | `crossvid` | `crossvid-debug` |
| `CROSSVID_EXPERIMENT_NAME` | wandb experiment name | `agent_4b` | `agent` |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0,1,2,3,4,5,6,7` | same |
| `WANDB_API_KEY` | wandb API key (optional) | — | — |
| `CROSSVID_DEBUG_LOG` | Path for CrossVid reward debug log (optional) | no logging | no logging |

Tool configuration (API URL, API key, etc.) is in `examples/sglang_multiturn/config/tool_config/video_analysis_tools.yaml`.

## Reward design

- **Correctness reward**: Computed in `verl/utils/reward_score/crossvid.py` by `compute_score`; 1.0 when the final answer matches the ground truth, 0.0 otherwise.
- **Format reward**: If every tool call in a query satisfies the format (valid tool name and valid JSON arguments), an extra **0.1** is added.
