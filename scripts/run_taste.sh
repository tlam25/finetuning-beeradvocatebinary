#!/usr/bin/env bash
# Train aspect = appearance.
#
# Biến môi trường có thể override (hoặc đặt trong .env):
#     MODEL            bart | t5     (default: bart)
#     NPROC_PER_NODE   số GPU        (default: 2)
#     PUSH_TO_HUB      1 | 0         (default: 1, push checkpoint lên HF Hub)
#     HUB_USERNAME     HF username   (bắt buộc nếu PUSH_TO_HUB=1)
#     WANDB_API_KEY    key wandb     (từ .env)
#     HF_TOKEN         HF token      (từ .env, cần Write access)
#
# Mặc định đọc .env ở thư mục gốc của repo. Ví dụ chạy:
#     cp .env.example .env && vim .env        # điền 3 giá trị
#     bash scripts/run_appearance.sh          # BART, seed 999
#     MODEL=t5 bash scripts/run_appearance.sh # T5, seed 999
#     NPROC_PER_NODE=4 bash scripts/run_appearance.sh

set -euo pipefail

# =========== Load .env (nếu có) ===========
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# =========== Chọn model ===========
MODEL="${MODEL:-bart}"
case "${MODEL}" in
    bart)
        TRAIN_SCRIPT="src/train_bart.py"
        CHECKPOINT="facebook/bart-large"
        WANDB_PROJECT="BeerAdvocate-BartLarge-LoRA-4aspects"
        # BART-large OK với bf16 (A100/H100) hoặc fp16 (V100/T4).
        PRECISION_FLAG="--bf16"
        ;;
    t5)
        TRAIN_SCRIPT="src/train_t5.py"
        CHECKPOINT="t5-large"
        WANDB_PROJECT="BeerAdvocate-T5Large-LoRA-4aspects"
        # T5-large fp16 hay NaN → bf16 nếu GPU hỗ trợ, ngược lại fp32.
        PRECISION_FLAG="--bf16"
        # PRECISION_FLAG=""   # fp32 cho T4/V100 để tránh NaN
        ;;
    *)
        echo "Unknown MODEL='${MODEL}'. Expected 'bart' or 't5'." >&2
        exit 1
        ;;
esac

# =========== Hyperparameters chung ===========
ASPECT="taste"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

HF_DATASET_REPO="tlam25/BeerAdvocate-binary"
OUTPUT_ROOT="outputs/${MODEL}"

TRAIN_SUBSET_SIZE=2000
MAX_INPUT_LENGTH=384
MAX_TARGET_LENGTH=4

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

NUM_TRAIN_EPOCHS=10
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=20
SAVE_TOTAL_LIMIT=2
DATALOADER_NUM_WORKERS=2
NUM_BEAMS=4

SEEDS=(999)
# SEEDS=(999 2025 42)

# =========== HF Hub push setup ===========
PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
HUB_USERNAME="${HUB_USERNAME:-}"

# =========== Launch ===========
for SEED in "${SEEDS[@]}"; do
    # Compose hub_model_id per run (aspect × seed × model)
    PUSH_FLAGS=""
    if [ "${PUSH_TO_HUB}" = "1" ]; then
        if [ -z "${HUB_USERNAME}" ]; then
            echo "(!) PUSH_TO_HUB=1 nhưng HUB_USERNAME chưa set. Cập nhật .env hoặc tắt PUSH_TO_HUB." >&2
            exit 1
        fi
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "(!) PUSH_TO_HUB=1 nhưng HF_TOKEN chưa set (cần Write access). Cập nhật .env." >&2
            exit 1
        fi
        HUB_MODEL_ID="${HUB_USERNAME}/beeradv-${MODEL}-${ASPECT}-seed-${SEED}"
        PUSH_FLAGS="--push_to_hub --hub_model_id ${HUB_MODEL_ID} --hub_strategy checkpoint"
    fi

    echo "==========================================================="
    echo ">>> aspect=${ASPECT}  model=${MODEL}  seed=${SEED}  gpus=${NPROC_PER_NODE}"
    echo ">>> checkpoint=${CHECKPOINT}  precision=${PRECISION_FLAG:-fp32}"
    if [ -n "${PUSH_FLAGS}" ]; then
        echo ">>> push → ${HUB_MODEL_ID}"
    else
        echo ">>> push: disabled"
    fi
    echo "==========================================================="

    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${TRAIN_SCRIPT}" \
        --aspect "${ASPECT}" \
        --seed "${SEED}" \
        --hf_dataset_repo "${HF_DATASET_REPO}" \
        --train_subset_size "${TRAIN_SUBSET_SIZE}" \
        --checkpoint "${CHECKPOINT}" \
        --max_input_length "${MAX_INPUT_LENGTH}" \
        --max_target_length "${MAX_TARGET_LENGTH}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --num_beams "${NUM_BEAMS}" \
        --output_root "${OUTPUT_ROOT}" \
        --wandb_project "${WANDB_PROJECT}" \
        ${PRECISION_FLAG} \
        ${PUSH_FLAGS}
done
