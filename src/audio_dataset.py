import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import soundfile as sf
import torchaudio.transforms as T
import torchaudio.functional as TAF


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LA_ROOT = os.path.join(PROJECT_ROOT, "data/raw/archive/LA/LA")
SPLIT_DIRS = {
    "train": os.path.join(LA_ROOT, "ASVspoof2019_LA_train/flac"),
    "dev":   os.path.join(LA_ROOT, "ASVspoof2019_LA_dev/flac"),
    "eval":  os.path.join(LA_ROOT, "ASVspoof2019_LA_eval/flac"),
}
PROTOCOL_FILES = {
    "train": os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"),
}

SAMPLE_RATE = 16000
N_MELS     = 80
N_FFT      = 512
HOP_LENGTH = 160
MAX_FRAMES = 300   # ~3 seconds at 16kHz/160 hop


def parse_protocol(protocol_path: str) -> list[tuple[str, int]]:
    """Returns list of (file_id, label) where label: bonafide=0, spoof=1."""
    items = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            tag = parts[-1]  # 'bonafide' or 'spoof'
            label = 0 if tag == "bonafide" else 1
            items.append((file_id, label))
    return items


def load_waveform(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T)                              # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)                     # (1, T)
    if sr != target_sr:
        wav = TAF.resample(wav, sr, target_sr)
    return wav  # (1, T)


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """wav: (1, T) → log-mel spectrogram (1, N_MELS, MAX_FRAMES), padded/truncated."""
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel = mel_transform(wav)                     # (1, N_MELS, T_frames)
    log_mel = torch.log(mel + 1e-9)             # log scale

    # Pad or truncate time axis to MAX_FRAMES
    T_frames = log_mel.shape[2]
    if T_frames < MAX_FRAMES:
        pad = MAX_FRAMES - T_frames
        log_mel = torch.nn.functional.pad(log_mel, (0, pad))
    else:
        log_mel = log_mel[:, :, :MAX_FRAMES]

    # Normalize per-sample
    mean = log_mel.mean()
    std  = log_mel.std() + 1e-9
    log_mel = (log_mel - mean) / std

    return log_mel  # (1, N_MELS, MAX_FRAMES)


class ASVspoofDataset(Dataset):
    def __init__(self, split: str):
        """split: 'train', 'dev', or 'eval'"""
        assert split in SPLIT_DIRS, f"split must be one of {list(SPLIT_DIRS)}"
        self.flac_dir = SPLIT_DIRS[split]
        self.items = parse_protocol(PROTOCOL_FILES[split])
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_id, label = self.items[idx]
        path = os.path.join(self.flac_dir, f"{file_id}.flac")
        wav = load_waveform(path)
        logmel = wav_to_logmel(wav)
        return logmel, torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self) -> dict:
        labels = [lbl for _, lbl in self.items]
        return {0: labels.count(0), 1: labels.count(1)}


def get_audio_weighted_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    counts = dataset.get_class_counts()
    n_bonafide = counts.get(0, 1)
    n_spoof    = counts.get(1, 1)
    total = n_bonafide + n_spoof
    w_bonafide = total / (2 * n_bonafide)
    w_spoof    = total / (2 * n_spoof)
    weights = [
        w_bonafide if item[1] == 0 else w_spoof
        for item in dataset.items
    ]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_audio_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    train_ds = ASVspoofDataset("train")
    dev_ds   = ASVspoofDataset("dev")
    eval_ds  = ASVspoofDataset("eval")

    sampler = get_audio_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, dev_loader, eval_loader, train_ds


if __name__ == "__main__":
    train_loader, dev_loader, eval_loader, train_ds = get_audio_dataloaders(batch_size=16, num_workers=2)
    counts = train_ds.get_class_counts()
    print(f"Train: bonafide={counts[0]}, spoof={counts[1]}")
    print(f"Train batches: {len(train_loader)} | Dev: {len(dev_loader)} | Eval: {len(eval_loader)}")
    specs, labels = next(iter(train_loader))
    print(f"Spectrogram shape: {specs.shape}  Labels: {labels[:8]}")
