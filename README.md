# BeerAdvocate × LoRA fine-tuning (BART-large / T5-large, multi-GPU)

Cấu trúc:

```
beer_bart_lora/
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

### Setup trên máy mới

```bash
git clone <repo-url>
cd finetuning-beeradvocatebinary

# 1. Cài env
pip install -r requirements.txt

# 2. Tạo .env và điền key
cp .env.example .env
vim .env   # điền WANDB_API_KEY, HF_TOKEN, HUB_USERNAME
```

## Chạy 1 aspect

```bash
# BART-large (mặc định)
bash scripts/run_appearance.sh

# T5-large
MODEL=t5 bash scripts/run_appearance.sh

# 4 GPU
NPROC_PER_NODE=4 bash scripts/run_appearance.sh

# T5 trên 4 GPU, chạy aspect aroma
NPROC_PER_NODE=4 MODEL=t5 bash scripts/run_aroma.sh

# Tắt push lên HF (chỉ save local)
PUSH_TO_HUB=0 bash scripts/run_appearance.sh
```

Mặc định chỉ chạy `SEED=999`. Mở file .sh và bỏ comment dòng `SEEDS=(999 2025 42)` để chạy đủ 3 seed.

## Checkpoint trên HuggingFace Hub

Khi `PUSH_TO_HUB=1` (mặc định), checkpoint được đẩy lên HF Hub mỗi lần Trainer
save (= mỗi epoch). Mỗi run có repo riêng:

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

3. **Muốn xoá sạch và train lại từ đầu**: thêm `--overwrite` vào lệnh .sh
   (hoặc chạy `torchrun` trực tiếp với cờ này), đồng thời xoá repo HF thủ công
   nếu muốn.

## Chọn precision

Mặc định trong .sh là `--bf16`. Thay `PRECISION_FLAG` trong file .sh theo GPU:

| GPU | BART-large | T5-large |
|---|---|---|
| A100/H100/RTX 30+ | `--bf16` (mặc định) | `--bf16` (mặc định) |
| V100 | `--fp16` | `""` (fp32) — fp16 hay NaN |
| T4 | `--fp16` | `""` (fp32) — fp16 hay NaN |

**Lưu ý T5-large**: T5 fp16 nổi tiếng không ổn định (hay ra NaN loss). Với
T4/V100, bỏ trống `PRECISION_FLAG=""` để chạy fp32 — chậm nhưng không NaN.

## Output local

Mỗi run lưu tại `outputs/{model}/{aspect}-seed-{seed}/`:
- `adapter/` – LoRA adapter + tokenizer.
- `results.json` – metric dev/test + toàn bộ hyperparameters đã dùng.
- `checkpoints/` – bị xoá sau khi train xong để tiết kiệm ổ đĩa (thêm
  `--no_cleanup_checkpoints` trong .sh để giữ lại). Hub repo vẫn có bản
  backup đầy đủ.

## Khác biệt BART vs T5 trong code

| | `train_bart.py` | `train_t5.py` |
|---|---|---|
| Default checkpoint | `facebook/bart-large` | `t5-large` |
| LoRA target modules | `["q_proj", "v_proj"]` | `["q", "v"]` |
| Default wandb project | `BeerAdvocate-BartLarge-LoRA-4aspects-3runs` | `BeerAdvocate-T5Large-LoRA-4aspects-3runs` |

Phần còn lại hoàn toàn giống nhau.

## Ghi chú multi-GPU

- DDP tự bật khi launch qua `torchrun --nproc_per_node=N`.
  Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × N`.
  Với default và 2 GPU: `2 × 8 × 2 = 32`.
- Checkpoint chỉ ghi trên rank 0 (Trainer xử lý sẵn). Khi resume với khác số
  GPU, HF Trainer vẫn load được.
- Gradient checkpointing bật mặc định với `use_reentrant=False`.
- `model.enable_input_require_grads()` gọi trước khi wrap PEFT để gradient flow
  được qua embedding frozen tới LoRA layers.

## Troubleshooting

**Không push được lên HF Hub** — kiểm tra `HF_TOKEN` trong `.env` có **Write
access** không (không phải Read). Tạo token mới tại
https://huggingface.co/settings/tokens và chọn "Write".

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
`PRECISION_FLAG=""` (fp32) hoặc `--bf16` nếu GPU hỗ trợ.

**`sentencepiece` thiếu khi load T5 tokenizer** — đã có trong `requirements.txt`,
chỉ cần `pip install -r requirements.txt`.
