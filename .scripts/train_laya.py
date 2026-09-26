#!/usr/bin/env python3
"""
scripts/train_laya.py
---------------------
Standalone training script for fine-tuning Laya System 1 decision engine.
Designed for execution on free cloud GPUs (Google Colab / Kaggle T4 GPU).

Features:
- On-the-fly tokenization (reads transparent laya_train.jsonl and laya_val.jsonl).
- Automatic mixed precision (AMP fp16) on CUDA.
- Step-wise holdout validation with early stopping and best_checkpoint/ saving.
- Grid-search temperature calibration minimizing Expected Calibration Error (ECE).
- Exports model.safetensors, tokenizer, rl_agent_config.json, and laya-techjob.tar.gz.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_laya")


def compute_ece(probs: List[float], preds: List[int], targets: List[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    total = len(targets)
    if total == 0:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Gather samples in this confidence bin
        bin_indices = [
            idx for idx, p in enumerate(probs)
            if (bin_lower <= p < bin_upper) or (i == n_bins - 1 and bin_lower <= p <= bin_upper)
        ]

        if not bin_indices:
            continue

        bin_acc = sum(1 for idx in bin_indices if preds[idx] == targets[idx]) / len(bin_indices)
        bin_conf = sum(probs[idx] for idx in bin_indices) / len(bin_indices)
        bin_weight = len(bin_indices) / total

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def grid_search_temperature(logits: Any, targets: List[int]) -> Tuple[float, float, float]:
    """Find scalar temperature T in [0.1, 5.0] that minimizes ECE on validation logits."""
    import numpy as np

    logits_arr = np.array(logits, dtype=np.float64)
    best_t = 1.0
    best_ece = 1.0

    # Grid search candidate temperatures
    temperatures = np.arange(0.1, 5.05, 0.05)
    for t in temperatures:
        scaled_logits = logits_arr / t
        # Stable softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs_all = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        preds = np.argmax(probs_all, axis=-1).tolist()
        max_probs = np.max(probs_all, axis=-1).tolist()

        ece = compute_ece(max_probs, preds, targets)
        if ece < best_ece:
            best_ece = ece
            best_t = float(t)

    # Compute uncalibrated ECE (T=1.0)
    exp_uncal = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs_uncal = exp_uncal / np.sum(exp_uncal, axis=-1, keepdims=True)
    uncal_ece = compute_ece(np.max(probs_uncal, axis=-1).tolist(), np.argmax(probs_uncal, axis=-1).tolist(), targets)

    return best_t, best_ece, uncal_ece


def run_training(args: argparse.Namespace) -> None:
    """Execute the fine-tuning and calibration workflow."""
    logger.info("Initializing Laya fine-tuning pipeline...")
    logger.info(f"Training data: {args.train_file}")
    logger.info(f"Validation data: {args.val_file}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")

    # Check for dry-run or mock mode
    if args.dry_run:
        logger.info("[DRY-RUN] Executing dry-run verification mode.")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_type": "laya-multilingual",
            "base_model": args.base_model,
            "calibrated_temperature": 1.05,
            "best_val_loss": 0.285,
            "best_val_accuracy": 0.942,
            "calibrated_ece": 0.038,
            "uncalibrated_ece": 0.092,
            "tasks": [
                "match_scoring_skill",
                "match_scoring_seniority",
                "match_scoring_recruiter_fit",
                "dedup_verification",
                "role_classification",
                "section_parsing",
                "field_mapping",
            ],
            "dry_run": True,
        }

        with open(out_dir / "rl_agent_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Create dummy weights and tokenizer markers for verification
        (out_dir / "model.safetensors").touch()
        (out_dir / "tokenizer_config.json").touch()

        # Create tarball bundle
        tar_path = out_dir.parent / "laya-techjob.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname="laya-techjob")

        logger.info(f"[DRY-RUN] Success! Created mock bundle at {tar_path}")
        return

    # Real training flow with PyTorch and Transformers
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        logger.error(
            f"Required deep learning libraries not found: {e}.\n"
            f"To train, run on Google Colab / Kaggle with GPU, or install with:\n"
            f"  pip install torch transformers datasets safetensors accelerate"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute target: {device} (CUDA available: {torch.cuda.is_available()})")

    # Load datasets from JSONL
    train_records: List[Dict[str, Any]] = []
    with open(args.train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    val_records: List[Dict[str, Any]] = []
    with open(args.val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    logger.info(f"Loaded {len(train_records)} training records and {len(val_records)} validation records.")

    # Initialize Tokenizer
    logger.info(f"Loading tokenizer: {args.base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as err:
        logger.warning(f"Could not load {args.base_model} directly ({err}). Falling back to 'bert-base-multilingual-cased'.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    class LayaDataset(Dataset):
        def __init__(self, records: List[Dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = self.records[idx]
            # Construct clear prompt representation
            options_text = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(item["options"]))
            prompt = (
                f"Context:\n{item['state']}\n\n"
                f"Instruction: {item['instruction']}\n"
                f"Candidate Options:\n{options_text}"
            )
            return {
                "text": prompt,
                "label": int(item["label"]),
                "task": item["task"],
            }

    train_ds = LayaDataset(train_records)
    val_ds = LayaDataset(val_records)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    # Determine max num_labels across all tasks (field_mapping has up to 15)
    num_labels = max(len(r["options"]) for r in train_records + val_records)
    logger.info(f"Initializing classification model with num_labels={num_labels}...")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=num_labels,
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=num_labels,
        )

    model.to(device)

    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.output_dir)
    best_dir = out_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    logger.info("Beginning training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Evaluation check
            if global_step % args.eval_steps == 0:
                model.eval()
                val_loss = 0.0
                correct = 0
                val_total = 0
                all_val_logits: List[List[float]] = []
                all_val_targets: List[int] = []

                with torch.no_grad():
                    for vbatch in val_loader:
                        v_ids = vbatch["input_ids"].to(device)
                        v_mask = vbatch["attention_mask"].to(device)
                        v_lbls = vbatch["labels"].to(device)

                        with torch.cuda.amp.autocast(enabled=use_amp):
                            v_out = model(input_ids=v_ids, attention_mask=v_mask, labels=v_lbls)
                            val_loss += v_out.loss.item() * len(v_lbls)

                        logits = v_out.logits.cpu().tolist()
                        targets = v_lbls.cpu().tolist()
                        all_val_logits.extend(logits)
                        all_val_targets.extend(targets)

                        preds = [max(range(len(l)), key=lambda i: l[i]) for l in logits]
                        correct += sum(1 for p, t in zip(preds, targets) if p == t)
                        val_total += len(targets)

                avg_val_loss = val_loss / max(1, val_total)
                val_acc = correct / max(1, val_total)
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {global_step} | "
                    f"Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    logger.info(f"--> New best model! Saving checkpoint to {best_dir}")
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                else:
                    patience_counter += 1
                    logger.info(f"Holdout loss did not improve. Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        logger.warning("Early stopping triggered! Restoring best checkpoint.")
                        break

                model.train()

        if patience_counter >= args.patience:
            break

    # Load best checkpoint for calibration
    logger.info("Loading best checkpoint for post-training temperature calibration...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(best_dir)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.warning(f"Could not reload best checkpoint: {e}. Calibrating final state.")

    # Collect final validation logits for grid-search calibration
    all_val_logits = []
    all_val_targets = []
    with torch.no_grad():
        for vbatch in val_loader:
            v_ids = vbatch["input_ids"].to(device)
            v_mask = vbatch["attention_mask"].to(device)
            v_lbls = vbatch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                v_out = model(input_ids=v_ids, attention_mask=v_mask)
            all_val_logits.extend(v_out.logits.cpu().tolist())
            all_val_targets.extend(v_lbls.cpu().tolist())

    logger.info("Running Grid-Search Temperature Calibration on holdout set...")
    best_t, calibrated_ece, uncalibrated_ece = grid_search_temperature(all_val_logits, all_val_targets)
    logger.info(
        f"Calibration Results: Optimal Temperature T={best_t:.2f} | "
        f"Uncalibrated ECE: {uncalibrated_ece:.4f} -> Calibrated ECE: {calibrated_ece:.4f}"
    )

    # Save final artifacts to output_dir
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    config = {
        "model_type": "laya-multilingual",
        "base_model": args.base_model,
        "calibrated_temperature": best_t,
        "best_val_loss": best_val_loss,
        "calibrated_ece": calibrated_ece,
        "uncalibrated_ece": uncalibrated_ece,
        "tasks": [
            "match_scoring_skill",
            "match_scoring_seniority",
            "match_scoring_recruiter_fit",
            "dedup_verification",
            "role_classification",
            "section_parsing",
            "field_mapping",
        ],
    }

    with open(out_dir / "rl_agent_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Bundle into single archive for easy download from Colab/Kaggle
    tar_path = out_dir.parent / "laya-techjob.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname="laya-techjob")

    logger.info(f"Model exported successfully to {out_dir}")
    logger.info(f"Ready-to-deploy archive created at: {tar_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate Laya System 1 decision engine.")
    parser.add_argument("--train-file", type=Path, default=Path("data/training/laya_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/holdout/laya_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/laya-techjob"))
    parser.add_argument("--base-model", type=str, default="convaiinnovations/laya-multilingual")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Execute mock training loop without GPU")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
