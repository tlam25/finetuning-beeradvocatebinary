# BeerAdvocate × LoRA fine-tuning (BART-large / T5-large, single & multi-GPU)

Cấu trúc:

```
finetuning-beeradvocatebinary/
├── src/
│   ├── train_bart.py          # BART-large + LoRA (target modules: q_proj, v_proj)
│   └── train_t5.py            # T5-large   + LoRA (target modules: q, v)
├── scripts/
│   ├── run_appearance.sh      # Mỗi .sh hỗ trợ cả 2 model qua biến MODEL
│   ├── run_aroma.sh
│   ├── run_palate.sh
│   └── run_taste.sh
├── .env.example               # Template cho .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup trên máy mới

```bash
git clone <repo-url>
cd finetuning-beeradvocatebinary

# 1. Cài env
pip install -r requirements.txt

# 2. Tạo .env và điền key
cp .env.example .env
vim .env   # điền WANDB_API_KEY, HF_TOKEN, HUB_USERNAME
```

## Chạy với 1 GPU (NVIDIA L4 / g2-standard-4)

Đây là cấu hình mặc định cho deployment hiện tại: 1× L4 24GB, 4 vCPU, 16GB RAM.

### Cách 1 (recommended): vẫn dùng `.sh` với `NPROC_PER_NODE=1`

`torchrun --nproc_per_node=1` chạy ngon trên 1 GPU, overhead init DDP rất nhỏ, và bạn được toàn bộ logic auto-load `.env` + build `hub_model_id`:

```bash
# Khởi động tmux session để job không chết khi mất SSH
tmux new -s train

# Trong tmux:
cd finetuning-beeradvocatebinary
NPROC_PER_NODE=1 bash scripts/run_appearance.sh

# Detach: Ctrl+b rồi nhấn d
# Reattach sau: tmux attach -t train
```

### Cách 2: chạy `python3` trực tiếp (không qua torchrun)

Khi chạy trực tiếp `python3`, các biến trong `.env` không tự convert thành CLI flag. Phải tự thêm `--bf16`, `--push_to_hub`, `--hub_model_id`:

```bash
tmux new -s train

# Load .env để HF_TOKEN, WANDB_API_KEY available trong shell
set -a && source .env && set +a

python3 src/train_bart.py \
    --aspect appearance \
    --seed 999 \
    --bf16 \
    --push_to_hub \
    --hub_model_id "tlam25/beeradv-bart-appearance-seed-999"
```

Để chạy 4 aspect liên tiếp trong 1 tmux session:

```bash
for ASPECT in appearance aroma palate taste; do
    python3 src/train_bart.py \
        --aspect "$ASPECT" \
        --seed 999 \
        --bf16 \
        --push_to_hub \
        --hub_model_id "tlam25/beeradv-bart-${ASPECT}-seed-999"
done
```

### tmux cheatsheet

| Mục đích | Lệnh |
|---|---|
| Tạo session mới | `tmux new -s train` |
| Detach (giữ chạy) | `Ctrl+b` rồi `d` |
| List session | `tmux ls` |
| Reattach | `tmux attach -t train` |
| Kill session | `tmux kill-session -t train` |
| Scroll trong tmux | `Ctrl+b` rồi `[` (sau đó dùng PageUp/Down, `q` để thoát) |

### Effective batch size với 1 GPU

Default config trong `.sh`:

```
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

→ Với 1 GPU: effective batch = `2 × 8 × 1 = 16`
→ Với 2 GPU (notebook gốc): `2 × 8 × 2 = 32`

Nếu muốn match đúng effective batch = 32 của notebook gốc khi chạy 1 GPU, sửa
trong `.sh`:

```bash
GRADIENT_ACCUMULATION_STEPS=16
```

Với binary classification + 2000 samples train, effective batch 16 cũng OK.

## Chạy với multi-GPU (nếu nâng cấp lên A100, H100 v.v.)

```bash
# BART-large mặc định, 1 GPU
bash scripts/run_appearance.sh

# T5-large
MODEL=t5 bash scripts/run_appearance.sh

# 4 GPU
NPROC_PER_NODE=4 bash scripts/run_appearance.sh

# T5 trên 4 GPU, aspect aroma
NPROC_PER_NODE=4 MODEL=t5 bash scripts/run_aroma.sh

# Tắt push lên HF (chỉ save local)
PUSH_TO_HUB=0 bash scripts/run_appearance.sh
```

Mặc định chỉ chạy `SEED=999`. Mở file `.sh` và bỏ comment dòng
`SEEDS=(999 2025 42)` để chạy đủ 3 seed.

## Checkpoint trên HuggingFace Hub

Khi `PUSH_TO_HUB=1` (mặc định trong `.sh`), checkpoint được đẩy lên HF Hub mỗi
lần Trainer save (= mỗi epoch). Mỗi run có repo riêng:

```
{HUB_USERNAME}/beeradv-{model}-{aspect}-seed-{seed}
```

Ví dụ: `tlam25/beeradv-bart-appearance-seed-999`.

Cấu trúc repo trên Hub:

```
repo/
├── last-checkpoint/            # checkpoint mới nhất (đầy đủ optimizer, scheduler, RNG state)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── rng_state_*.pth
├── adapter/                    # adapter final (push sau khi train xong)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── adapter_model.safetensors   # mirror của adapter/ ở root (Trainer tự push)
├── adapter_config.json
└── results.json                # metric dev/test
```

### Resume sau khi server bị kill / bị wipe ổ đĩa

1. **Nếu local còn checkpoint**: chạy lại đúng lệnh cũ. Trainer auto-detect
   `outputs/{model}/{aspect}-seed-{seed}/checkpoints/checkpoint-*` và resume.
2. **Nếu local bị wipe sạch nhưng HF Hub có last-checkpoint**: cũng chạy lại
   đúng lệnh cũ. Script tự `snapshot_download(allow_patterns="last-checkpoint/*")`
   từ Hub vào local, rồi resume từ đó. Logic này nằm trong `train_*.py`, không
   cần thao tác tay.
3. **Muốn xoá sạch và train lại từ đầu**: thêm `--overwrite` vào lệnh `.sh`
   (hoặc chạy `python3` trực tiếp với cờ này), đồng thời xoá repo HF thủ công
   nếu muốn.

## Chọn precision

Mặc định trong `.sh` là `--bf16`. Bảng GPU:

| GPU | BART-large | T5-large |
|---|---|---|
| **L4 (Ada, sm_89)** ⭐ deployment hiện tại | `--bf16` (mặc định) | `--bf16` (mặc định) |
| A100/H100 | `--bf16` (mặc định) | `--bf16` (mặc định) |
| RTX 30/40 series | `--bf16` (mặc định) | `--bf16` (mặc định) |
| V100 | `--fp16` | `""` (fp32) — fp16 hay NaN |
| T4 | `--fp16` | `""` (fp32) — fp16 hay NaN |

**Lưu ý T5-large**: T5 fp16 nổi tiếng không ổn định (hay ra NaN loss). Với
T4/V100, bỏ trống `PRECISION_FLAG=""` để chạy fp32 — chậm nhưng không NaN. L4
hỗ trợ bf16 native nên không gặp vấn đề này.

## Output local

Mỗi run lưu tại `outputs/{model}/{aspect}-seed-{seed}/`:

- `adapter/` – LoRA adapter + tokenizer.
- `results.json` – metric dev/test + toàn bộ hyperparameters đã dùng.
- `checkpoints/` – bị xoá sau khi train xong để tiết kiệm ổ đĩa (thêm
  `--no_cleanup_checkpoints` trong `.sh` để giữ lại). Hub repo vẫn có bản
  backup đầy đủ.

## Khác biệt BART vs T5 trong code

| | `train_bart.py` | `train_t5.py` |
|---|---|---|
| Default checkpoint | `facebook/bart-large` | `t5-large` |
| LoRA target modules | `["q_proj", "v_proj"]` | `["q", "v"]` |
| Default wandb project | `BeerAdvocate-BartLarge-LoRA-4aspects` | `BeerAdvocate-T5Large-LoRA-4aspects` |

Phần còn lại hoàn toàn giống nhau.

## Ghi chú multi-GPU

- DDP tự bật khi launch qua `torchrun --nproc_per_node=N` (N > 1).
  Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × N`.
  Với default và 2 GPU: `2 × 8 × 2 = 32`.
- Checkpoint chỉ ghi trên rank 0 (Trainer xử lý sẵn). Khi resume với khác số
  GPU, HF Trainer vẫn load được.
- Gradient checkpointing bật mặc định với `use_reentrant=False`.
- `model.enable_input_require_grads()` gọi trước khi wrap PEFT để gradient flow
  được qua embedding frozen tới LoRA layers.

## Ghi chú về g2-standard-4 (1× L4, 4 vCPU, 16GB RAM)

- 16GB RAM hơi chật khi load BART-large + dataset + tokenizer. Theo dõi `htop`
  hoặc `free -h` trong 1-2 epoch đầu để chắc không bị swap.
- 4 vCPU → giữ `DATALOADER_NUM_WORKERS=2` (đừng tăng, sẽ thrash CPU).
- L4 24GB VRAM dư cho BART-large + LoRA + max_len=384. Có thể tăng
  `per_device_train_batch_size` lên 4 nếu muốn (giảm `grad_accum` tương ứng).
- T5-large trên L4 cũng OK nhưng sát hơn — nếu OOM, giữ nguyên batch=2 và
  tăng `grad_accum`.

## Troubleshooting

**Job chết khi đóng terminal / mất SSH** — luôn chạy trong `tmux` (xem
section "Chạy với 1 GPU" ở trên).

**Không push được lên HF Hub** — kiểm tra `HF_TOKEN` trong `.env` có **Write
access** không (không phải Read). Tạo token mới tại
https://huggingface.co/settings/tokens và chọn "Write".

**Chạy `python3 src/train_bart.py` trực tiếp nhưng không thấy push lên Hub** —
khi chạy trực tiếp (không qua `.sh`), các biến `.env` không tự thành CLI flag.
Phải tự thêm `--push_to_hub --hub_model_id <id>` vào lệnh.

**Chạy `python3` trực tiếp nhưng training cực chậm** — nhiều khả năng quên
`--bf16` → đang chạy fp32. Default của `argparse` là `bf16=False`, chỉ `.sh`
mới auto bật.

**`ImportError: Found an incompatible version of torchao ... only versions above 0.16.0 are supported`**
```bash
pip uninstall -y torchao
```

**`ImportError: cannot import name 'EmbeddingParallel'`** — peft 0.18+ cần
transformers v5. `requirements.txt` đã pin đúng cặp `transformers<5` + `peft<0.18`;
nếu gặp lỗi thì `pip install -r requirements.txt --upgrade`.

**`TypeError: ... unexpected keyword argument 'save_safetensors'` (hoặc arg khác)** —
đang chạy transformers v5. Downgrade:
```bash
pip install "transformers>=4.46.0,<5.0.0"
```

**`RuntimeError: element 0 of tensors does not require grad`** — gradient
checkpointing cần `model.enable_input_require_grads()` trước khi wrap PEFT.
Code đã có fix này.

**T5 loss = NaN sau vài step** — vấn đề kinh điển của T5 + fp16. Đổi
`PRECISION_FLAG=""` (fp32) hoặc `--bf16` nếu GPU hỗ trợ. L4 hỗ trợ bf16 nên
không nên gặp lỗi này.

**`sentencepiece` thiếu khi load T5 tokenizer** — đã có trong `requirements.txt`,
chỉ cần `pip install -r requirements.txt`.

**OOM trên L4** — giảm `per_device_train_batch_size` xuống 1 và tăng
`gradient_accumulation_steps` lên gấp đôi để giữ effective batch.
