import os
import sys
import json
import argparse
import logging

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_dataset import get_audio_dataloaders
from src.audio_classifier import load_audio_model, predict_audio_batch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("audio_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "audio_eval.log"))
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def evaluate(args):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = load_audio_model(device=device)
    logger.info("LCNN audio model loaded")

    _, dev_loader, eval_loader, _ = get_audio_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
    )

    results = {}
    for split_name, loader in [("dev", dev_loader), ("eval", eval_loader)]:
        all_probs, all_labels = [], []
        for specs, labels in loader:
            probs = predict_audio_batch(specs, model, device)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc  = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec  = recall_score(all_labels, preds, zero_division=0)
        f1   = f1_score(all_labels, preds, zero_division=0)
        auc  = roc_auc_score(all_labels, all_probs)
        cm   = confusion_matrix(all_labels, preds).tolist()

        # Equal Error Rate approximation
        eer = _compute_eer(all_labels, all_probs)

        results[split_name] = {
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1_score":  round(f1,   4),
            "auc_roc":   round(auc,  4),
            "eer":       round(eer,  4),
            "confusion_matrix": cm,
            "n_samples": len(all_labels),
        }

        logger.info(f"=== {split_name.upper()} SET ===")
        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall   : {rec:.4f}")
        logger.info(f"F1-score : {f1:.4f}")
        logger.info(f"AUC-ROC  : {auc:.4f}")
        logger.info(f"EER      : {eer:.4f}  (lower = better)")
        logger.info(f"Confusion (bonafide=0, spoof=1):\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")

    out_path = os.path.join(LOG_DIR, "audio_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {out_path}")


def _compute_eer(labels, probs) -> float:
    """Approximate EER by finding threshold where FAR ≈ FRR."""
    import numpy as np
    thresholds = sorted(set(probs))
    best_eer = 1.0
    labels_arr = [int(l) for l in labels]
    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in probs]
        fp = sum(1 for l, p in zip(labels_arr, preds) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(labels_arr, preds) if l == 1 and p == 0)
        n_neg = labels_arr.count(0)
        n_pos = labels_arr.count(1)
        far = fp / n_neg if n_neg > 0 else 0
        frr = fn / n_pos if n_pos > 0 else 0
        eer_approx = (far + frr) / 2
        if abs(far - frr) < abs(best_eer - 0.5) + 0.5:
            best_eer = eer_approx
    return best_eer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
