#!/bin/bash
# CrossVid agent training (4B model): run from repo root.
# Set env vars to override paths (optional):
#   CROSSVID_DATA_FILE   - train/val parquet path
#   CROSSVID_MODEL_PATH  - actor model path (default: Qwen/Qwen3-4B)
#   CROSSVID_CRITIC_PATH - critic model path
#   CROSSVID_CONFIG_DIR  - config directory

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

cleanup() {
    echo "Caught signal, stopping training..."
    [ -n "$PID" ] && kill -SIGINT "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    echo "Stopping Ray..."
    ray stop 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

DATA_FILE="${CROSSVID_DATA_FILE:-$REPO_ROOT/examples/data_preprocess/format_data/crossvid_train_mini.parquet}"
MODEL_PATH="${CROSSVID_MODEL_PATH:-Qwen/Qwen3-4B}"
CRITIC_MODEL_PATH="${CROSSVID_CRITIC_PATH:-Qwen/Qwen-Tiny-Critic}"
CONFIG_PATH="${CROSSVID_CONFIG_DIR:-$REPO_ROOT/examples/sglang_multiturn/config}"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: data file not found: $DATA_FILE"
    echo "Set CROSSVID_DATA_FILE or place data at examples/data_preprocess/format_data/crossvid_train_mini.parquet"
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=crossvid_qwen3_grpo \
    data.train_files="$DATA_FILE" \
    data.val_files="$DATA_FILE" \
    trainer.val_before_train=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    trainer.n_gpus_per_node=8 \
    trainer.actor.optim.lr=1e-7 \
    algorithm.kl_ctrl.kl_coef=0.005 \
    trainer.project_name="${CROSSVID_PROJECT_NAME:-crossvid}" \
    trainer.experiment_name="${CROSSVID_EXPERIMENT_NAME:-agent_4b}" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    'trainer.logger=[console,wandb]' \
    trainer.total_epochs=2 \
    trainer.test_freq=10000 \
    trainer.save_freq=5 \
    critic.model.path="$CRITIC_MODEL_PATH" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.agent.num_workers=4 &

PID=$!
echo "Training started (4B, PID: $PID). Data: $DATA_FILE"
wait $PID
