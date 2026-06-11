import os
import sys
import time
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders, get_pos_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def build_model(pretrained: bool = True) -> nn.Module:
    model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=1)
    return model


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.squeeze(1).tolist())
            all_preds.extend(preds.squeeze(1).tolist())
            all_labels.extend(labels.detach().cpu().squeeze(1).tolist())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


def train(args):
    logger = setup_logger("vit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    csv_path = os.path.join(PROJECT_ROOT, "data/processed/dataset.csv")
    train_loader, val_loader, _, train_ds = get_dataloaders(
        csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_filter=args.dataset_filter,
        use_weighted_sampler=True,
    )

    model = build_model(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: ViT-Small/16 (pretrained ImageNet) | Params: {n_params:.1f}M")

    pos_weight = get_pos_weight(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc, val_f1, val_auc = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} auc={tr_auc:.4f} | "
            f"Val   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f} | "
            f"Time={elapsed:.0f}s"
        )

        record = dict(
            epoch=epoch,
            tr_loss=tr_loss, tr_acc=tr_acc, tr_f1=tr_f1, tr_auc=tr_auc,
            val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, val_auc=val_auc,
        )
        history.append(record)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "vit_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, ckpt_path)
            logger.info(f"  ✓ Best checkpoint saved (val_auc={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    hist_path = os.path.join(LOG_DIR, "vit_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")
    logger.info(f"History saved: {hist_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dataset_filter", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
