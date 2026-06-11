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
from src.dataset import get_dataloaders
from src.ensemble import load_ensemble, predict_proba

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ensemble_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "ensemble_eval.log"))
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def evaluate(args):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Optional per-model weights from CLI, else equal
    weights = None
    if args.weights:
        keys = ["xception", "efficientnet_b0", "vit"]
        vals = [float(x) for x in args.weights.split(",")]
        if len(vals) != 3:
            raise ValueError("--weights must be 3 comma-separated floats, e.g. 1.2,1.0,0.8")
        weights = dict(zip(keys, vals))
        logger.info(f"Custom weights: {weights}")

    models, weights = load_ensemble(weights=weights, device=device)
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

    csv_path = os.path.join(PROJECT_ROOT, "data/processed/dataset.csv")
    _, _, test_loader, _ = get_dataloaders(
        csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_filter=args.dataset_filter,
        use_weighted_sampler=False,
    )
    logger.info(f"Test batches: {len(test_loader)}")

    all_probs, all_labels = [], []

    for imgs, labels in test_loader:
        probs = predict_proba(imgs, models, weights, device)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    acc  = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec  = recall_score(all_labels, preds, zero_division=0)
    f1   = f1_score(all_labels, preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs)
    cm   = confusion_matrix(all_labels, preds).tolist()

    results = {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "auc_roc":   round(auc,  4),
        "confusion_matrix": cm,
        "weights_used": weights,
        "dataset_filter": args.dataset_filter,
        "n_samples": len(all_labels),
    }

    logger.info("=" * 50)
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1-score : {f1:.4f}")
    logger.info(f"AUC-ROC  : {auc:.4f}")
    logger.info(f"Confusion matrix (real=0, fake=1):\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")
    logger.info("=" * 50)

    out_path = os.path.join(LOG_DIR, "ensemble_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dataset_filter", type=str, default=None,
                   help="'FaceForensics++', 'Celeb-DF', 'DFDC', or None for all")
    p.add_argument("--weights", type=str, default=None,
                   help="Comma-separated weights for xception,efficientnet_b0,vit (e.g. '1.2,1.0,0.8')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
