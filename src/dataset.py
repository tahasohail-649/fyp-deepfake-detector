import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, split: str, dataset_filter: str = None):
        """
        Args:
            csv_path: path to dataset.csv
            split: 'train', 'val', or 'test'
            dataset_filter: optional — 'FaceForensics++', 'Celeb-DF', 'DFDC', or None for all
        """
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if dataset_filter:
            df = df[df["dataset"] == dataset_filter].reset_index(drop=True)
        self.df = df
        self.split = split
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(PROJECT_ROOT, row["path"])
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self):
        counts = self.df["label"].value_counts().to_dict()
        return {int(k): int(v) for k, v in counts.items()}


def get_weighted_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    counts = dataset.get_class_counts()
    n_real = counts.get(0, 1)
    n_fake = counts.get(1, 1)
    total = n_real + n_fake

    weight_real = total / (2 * n_real)
    weight_fake = total / (2 * n_fake)

    weights = [
        weight_real if int(dataset.df.iloc[i]["label"]) == 0 else weight_fake
        for i in range(len(dataset))
    ]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_pos_weight(dataset: DeepfakeDataset) -> torch.Tensor:
    counts = dataset.get_class_counts()
    n_real = counts.get(0, 1)
    n_fake = counts.get(1, 1)
    # BCEWithLogitsLoss pos_weight: weight for positive (fake=1) class
    # Use inverse ratio to up-weight the minority (real) class
    return torch.tensor([n_real / n_fake], dtype=torch.float32)


def get_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_filter: str = None,
    use_weighted_sampler: bool = True,
):
    train_ds = DeepfakeDataset(csv_path, "train", dataset_filter)
    val_ds = DeepfakeDataset(csv_path, "val", dataset_filter)
    test_ds = DeepfakeDataset(csv_path, "test", dataset_filter)

    sampler = get_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds


if __name__ == "__main__":
    csv_path = os.path.join(PROJECT_ROOT, "data/processed/dataset.csv")
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        csv_path, batch_size=32, num_workers=4
    )

    counts = train_ds.get_class_counts()
    pos_weight = get_pos_weight(train_ds)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")
    print(f"Class counts  : real={counts[0]}, fake={counts[1]}")
    print(f"pos_weight    : {pos_weight.item():.4f}")

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape   : {imgs.shape}")
    print(f"Labels sample : {labels[:8]}")
    print(f"Label balance in batch — real: {(labels==0).sum()}, fake: {(labels==1).sum()}")
