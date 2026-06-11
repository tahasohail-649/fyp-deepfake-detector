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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_dataset import get_audio_dataloaders, ASVspoofDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("audio_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "audio_train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class MFM(nn.Module):
    """Max Feature Map activation — halves channels by max-pooling pairs."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    """
    Light CNN for anti-spoofing.
    Input: (B, 1, 80, 300) log-mel spectrogram
    Output: (B, 1) logit
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MFM(),                                       # → 32 ch
            nn.MaxPool2d(2, 2),                          # → 32, 40, 150

            nn.Conv2d(32, 64, kernel_size=1),
            MFM(),                                       # → 32 ch
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MFM(),                                       # → 48 ch
            nn.MaxPool2d(2, 2),                          # → 48, 20, 75
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, kernel_size=1),
            MFM(),                                       # → 48 ch
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MFM(),                                       # → 64 ch
            nn.MaxPool2d(2, 2),                          # → 64, 10, 37

            nn.Conv2d(64, 128, kernel_size=1),
            MFM(),                                       # → 64 ch
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MFM(),                                       # → 32 ch
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=1),
            MFM(),                                       # → 32 ch
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MFM(),                                       # → 32 ch
            nn.MaxPool2d(2, 2),                          # → 32, 5, 18 (approx)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            MFM(),                                       # → 32
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for specs, labels in loader:
            specs  = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model(specs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.squeeze(1).tolist())
            all_preds.extend(preds.squeeze(1).tolist())
            all_labels.extend(labels.detach().cpu().squeeze(1).tolist())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


def train(args):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, dev_loader, _, train_ds = get_audio_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )
    counts = train_ds.get_class_counts()
    logger.info(f"Train set: bonafide={counts[0]}, spoof={counts[1]}")

    model = LCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"LCNN params: {n_params:.2f}M")

    n_bonafide = counts.get(0, 1)
    n_spoof    = counts.get(1, 1)
    pos_weight = torch.tensor([n_bonafide / n_spoof], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            model, dev_loader, criterion, optimizer, device, is_train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} auc={tr_auc:.4f} | "
            f"Val   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f} | "
            f"Time={elapsed:.0f}s"
        )

        history.append(dict(
            epoch=epoch,
            tr_loss=tr_loss, tr_acc=tr_acc, tr_f1=tr_f1, tr_auc=tr_auc,
            val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, val_auc=val_auc,
        ))

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "audio_lcnn_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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

    hist_path = os.path.join(LOG_DIR, "audio_train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--num_workers", type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
