"""
Train BART-large + LoRA cho BeerAdvocate binary classification (1 aspect × 1 seed).

- Dataset: HF Hub  tlam25/BeerAdvocate-binary
    path bên trong repo: {aspect}/{aspect}_{split}_binary.csv
- Multi-GPU: khởi chạy bằng `torchrun --nproc_per_node=N` → HF Trainer tự bật DDP.
- Checkpoint mỗi epoch + resume tự động từ checkpoint cuối cùng nếu bị killed.
- Model selection: eval_loss (greater_is_better=False) để tiết kiệm thời gian.
- Wandb: chỉ init trên rank 0 (HF Trainer xử lý sẵn).

Ví dụ chạy:
    torchrun --nproc_per_node=2 src/train.py \
        --aspect appearance --seed 999 \
        --output_root /workspace/outputs \
        --wandb_project BeerAdvocate-BartLarge-LoRA
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import logging as hf_logging
from peft import LoraConfig, get_peft_model

hf_logging.set_verbosity_error()


# ----------------------------- CLI ----------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    # Bắt buộc
    p.add_argument("--aspect", required=True,
                   choices=["appearance", "aroma", "palate", "taste"])
    p.add_argument("--seed", type=int, required=True)

    # Data
    p.add_argument("--hf_dataset_repo", default="tlam25/BeerAdvocate-binary",
                   help="HuggingFace dataset repo id.")
    p.add_argument("--train_subset_size", type=int, default=2000,
                   help="Số sample train dùng (giống notebook gốc: 2000). -1 = dùng hết.")

    # Model
    p.add_argument("--checkpoint", default="facebook/bart-large")
    p.add_argument("--max_input_length", type=int, default=384)
    p.add_argument("--max_target_length", type=int, default=4)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Train
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    # I/O
    p.add_argument("--output_root", default="outputs",
                   help="Thư mục gốc, run sẽ nằm ở {output_root}/{aspect}-seed-{seed}.")
    p.add_argument("--overwrite", action="store_true",
                   help="Xoá run cũ trước khi train (mặc định: resume từ checkpoint).")
    p.add_argument("--cleanup_checkpoints", action="store_true", default=True,
                   help="Xoá thư mục checkpoint nặng sau khi train xong (giữ adapter + results).")
    p.add_argument("--no_cleanup_checkpoints", dest="cleanup_checkpoints",
                   action="store_false")

    # Wandb
    p.add_argument("--wandb_project",
                   default="BeerAdvocate-BartLarge-LoRA")
    p.add_argument("--wandb_run_name", default=None,
                   help="Mặc định: {aspect}-seed-{seed}.")
    p.add_argument("--report_to", default="wandb",
                   help="'wandb', 'none', hoặc csv nhiều target. Dùng 'none' để tắt log.")

    # HuggingFace Hub (backup checkpoint)
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push checkpoint lên HF Hub mỗi save. Yêu cầu env HF_TOKEN "
                        "và flag --hub_model_id.")
    p.add_argument("--hub_model_id", default=None,
                   help="Tên repo trên HF Hub, vd 'tlam25/beeradv-bart-appearance-seed-999'.")
    p.add_argument("--hub_private_repo", action="store_true", default=True,
                   help="Tạo repo private (mặc định).")
    p.add_argument("--hub_public_repo", dest="hub_private_repo",
                   action="store_false", help="Tạo repo public.")
    p.add_argument("--hub_strategy", default="checkpoint",
                   choices=["end", "every_save", "checkpoint", "all_checkpoints"],
                   help="Xem HF docs. 'checkpoint' = push last-checkpoint/ mỗi save, "
                        "cho phép resume từ Hub.")

    # Generation
    p.add_argument("--num_beams", type=int, default=4)

    return p.parse_args()


# ----------------------------- Main ----------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    import transformers, peft, datasets as _ds
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    if is_main_process:
        print(f"[env] transformers={transformers.__version__}  "
              f"peft={peft.__version__}  datasets={_ds.__version__}")

    run_name = args.wandb_run_name or f"{args.aspect}-seed-{args.seed}"
    output_dir = Path(args.output_root) / run_name

    # Overwrite: chỉ rank 0 xoá, các rank khác đợi
    if args.overwrite and output_dir.exists() and is_main_process:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Tokenizer ----- #
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # ----- Load + preprocess dataset ----- #
    label_col = f"{args.aspect}_binary_label"
    data_files = {
        "train": f"{args.aspect}/{args.aspect}_train_binary.csv",
        "dev":   f"{args.aspect}/{args.aspect}_dev_binary.csv",
        "test":  f"{args.aspect}/{args.aspect}_test_binary.csv",
    }
    raw = load_dataset(args.hf_dataset_repo, data_files=data_files)

    def preprocess_function(examples):
        formatted_inputs = [
            f"{args.aspect}: {t if t is not None else ''}"
            for t in examples["text"]
        ]
        model_inputs = tokenizer(
            formatted_inputs,
            max_length=args.max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels_text = [str(int(l)) for l in examples[label_col]]
        labels_tokenized = tokenizer(
            text_target=labels_text,
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        label_ids = labels_tokenized["input_ids"]
        bos_id = tokenizer.bos_token_id
        cleaned = []
        for seq in label_ids:
            if len(seq) > 0 and seq[0] == bos_id:
                seq = seq[1:] + [tokenizer.pad_token_id]
            cleaned.append(seq)
        model_inputs["labels"] = cleaned
        return model_inputs

    tokenized = raw.map(
        preprocess_function,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    if args.train_subset_size is not None and args.train_subset_size > 0:
        n = min(args.train_subset_size, len(tokenized["train"]))
        tokenized["train"] = tokenized["train"].select(range(n))

    if is_main_process:
        print({k: len(v) for k, v in tokenized.items()})

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,  # BART tự xử lý shift_tokens_right trong forward. Tránh
                     # truyền string (silent no-op với collator).
        padding=True,
        return_tensors="pt",
    )

    # ----- Metric ----- #
    def parse_pred(text):
        if text is None:
            return 0
        return 1 if "1" in text.strip() else 0

    def validate_tokens(tokens):
        tokens = np.array(tokens) if not isinstance(tokens, np.ndarray) else tokens
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return np.where(tokens < 0, pad_id, tokens)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = validate_tokens(predictions)
        labels = validate_tokens(labels)

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        y_pred = [parse_pred(p) for p in pred_texts]
        y_true = [parse_pred(l) for l in label_texts]

        acc = accuracy_score(y_true, y_pred)
        p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0,
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0,
        )
        return {
            "accuracy": acc,
            "f1_pos": f1_pos,
            "precision_pos": p_pos,
            "recall_pos": r_pos,
            "f1_macro": f1_macro,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
        }

    # ----- Wandb (rank 0 only; Trainer sẽ tự no-op trên rank khác) ----- #
    if args.report_to == "wandb" and is_main_process:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = args.aspect
        # Tên run cho wandb
        os.environ.setdefault("WANDB_NAME", run_name)

    # ----- Model + LoRA ----- #
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    model.resize_token_embeddings(len(tokenizer))

    # FIX quan trọng: gradient_checkpointing + LoRA cần cái này.
    # Embedding bị freeze → output requires_grad=False → checkpoint block không dựng
    # được gradient graph → backward crash. enable_input_require_grads đăng ký hook
    # trên embedding để set requires_grad=True cho output, gradient flow được qua LoRA.
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # ----- TrainingArguments ----- #
    report_to = [] if args.report_to in ("none", "", None) else args.report_to.split(",")

    # HF Hub push config
    push_hub = args.push_to_hub and bool(args.hub_model_id)
    if args.push_to_hub and not args.hub_model_id:
        if is_main_process:
            print("[warn] --push_to_hub bật nhưng không có --hub_model_id → tắt push.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        # use_reentrant=False là cách được khuyến nghị với DDP + find_unused_parameters
        # combos, và tránh warning "None of the inputs have requires_grad=True".
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=args.fp16,
        bf16=args.bf16,
        push_to_hub=push_hub,
        hub_model_id=args.hub_model_id if push_hub else None,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy if push_hub else "every_save",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
    )

    gen_config = GenerationConfig(
        min_length=1,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
        do_sample=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.generation_config = gen_config

    # ----- Resume logic ----- #
    ckpt_dir = output_dir / "checkpoints"

    def _has_local_ckpt():
        return ckpt_dir.exists() and any(
            p.name.startswith("checkpoint-") for p in ckpt_dir.iterdir()
        )

    has_ckpt = _has_local_ckpt()
    hf_last_ckpt = ckpt_dir / "last-checkpoint"

    # Nếu không có checkpoint local mà có push_to_hub + hub_model_id → thử restore từ Hub.
    if not has_ckpt and push_hub and not args.overwrite:
        if is_main_process:
            print(f">>> Không có checkpoint local. Thử restore từ HF Hub: {args.hub_model_id}")
            try:
                from huggingface_hub import snapshot_download
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=args.hub_model_id,
                    local_dir=str(ckpt_dir),
                    allow_patterns=["last-checkpoint/*"],
                )
                if hf_last_ckpt.exists():
                    print(f">>> Restored last-checkpoint từ Hub vào {hf_last_ckpt}")
                else:
                    print(">>> Hub repo chưa có last-checkpoint, sẽ train từ đầu.")
            except Exception as e:
                print(f">>> Không restore được từ Hub ({e}). Train từ đầu.")
        # Đồng bộ tất cả rank trước khi tiếp tục
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass

    if _has_local_ckpt() and not args.overwrite:
        if is_main_process:
            print(f">>> Resume từ checkpoint-* trong {ckpt_dir}")
        trainer.train(resume_from_checkpoint=True)
    elif hf_last_ckpt.exists() and not args.overwrite:
        if is_main_process:
            print(f">>> Resume từ {hf_last_ckpt} (đã restore từ HF Hub)")
        trainer.train(resume_from_checkpoint=str(hf_last_ckpt))
    else:
        trainer.train()

    # ----- Evaluate dev + test ----- #
    dev_result = trainer.evaluate(eval_dataset=tokenized["dev"], metric_key_prefix="eval")
    test_result = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")

    if is_main_process:
        print("\n>>> DEV :", dev_result)
        print(">>> TEST:", test_result)

        # Lưu adapter + tokenizer (local)
        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Push final adapter + results lên HF Hub (nếu bật)
        if push_hub:
            try:
                from huggingface_hub import HfApi, upload_folder
                api = HfApi()
                # Đảm bảo repo tồn tại
                api.create_repo(repo_id=args.hub_model_id,
                                private=args.hub_private_repo,
                                exist_ok=True)
                upload_folder(
                    folder_path=str(adapter_dir),
                    repo_id=args.hub_model_id,
                    path_in_repo="adapter",
                    commit_message=f"Final adapter — {run_name}",
                )
                print(f">>> Pushed final adapter → {args.hub_model_id}/adapter")
            except Exception as e:
                print(f">>> Push final adapter failed: {e}")

        # Lưu results
        def _to_py(d):
            return {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in d.items()
            }
        results = {
            "aspect": args.aspect,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "args": {k: v for k, v in vars(args).items()},
            "dev": _to_py(dev_result),
            "test": _to_py(test_result),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

        # Push results.json lên Hub
        if push_hub:
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(output_dir / "results.json"),
                    path_in_repo="results.json",
                    repo_id=args.hub_model_id,
                    commit_message=f"Results — {run_name}",
                )
                print(f">>> Pushed results.json → {args.hub_model_id}/results.json")
            except Exception as e:
                print(f">>> Push results.json failed: {e}")

        # Log summary vào wandb
        if "wandb" in report_to:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "final_dev_loss":             dev_result.get("eval_loss"),
                    "final_dev_f1_macro":         dev_result.get("eval_f1_macro"),
                    "final_test_loss":            test_result.get("test_loss"),
                    "final_test_accuracy":        test_result.get("test_accuracy"),
                    "final_test_f1_macro":        test_result.get("test_f1_macro"),
                    "final_test_precision_macro": test_result.get("test_precision_macro"),
                    "final_test_recall_macro":    test_result.get("test_recall_macro"),
                    "final_test_f1_pos":          test_result.get("test_f1_pos"),
                    "final_test_precision_pos":   test_result.get("test_precision_pos"),
                    "final_test_recall_pos":      test_result.get("test_recall_pos"),
                })
                wandb.finish()

        # Cleanup checkpoints nặng
        if args.cleanup_checkpoints and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

        print(f"\nSaved to: {output_dir}")
        print(f"   - {output_dir}/adapter/       (LoRA adapter + tokenizer)")
        print(f"   - {output_dir}/results.json   (metric dev/test)")


if __name__ == "__main__":
    main()
