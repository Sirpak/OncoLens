from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#walks a dir tree, retruns a Dataframe with at least path and label columns
def load_image_table(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.parent.name.lower()
        if label in CLASS_TO_INDEX:
            records.append({"image_path": str(path), "label": label})

    if not records:
        raise ValueError(
            "No LC25000 images were found. Point dataset.path to the folder containing class subfolders."
        )
    return pd.DataFrame(records)


def build_manifest(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    test_ratio = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio, stratify=temp_df["label"], random_state=seed
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


class LC25000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image) if self.transform else image
        return tensor, CLASS_TO_INDEX[row["label"]]


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def create_dataloaders(config: dict):
    dataset_cfg = config["dataset"]
    train_tfms, eval_tfms = build_transforms(int(dataset_cfg["image_size"]))
    manifest = build_manifest(
        load_image_table(dataset_cfg["path"]),
        val_size=float(dataset_cfg["val_size"]),
        test_size=float(dataset_cfg["test_size"]),
        seed=int(config["project"]["seed"]),
    )
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(dataset_cfg.get("num_workers", 2))
    datasets = {
        "train": LC25000Dataset(manifest[manifest["split"] == "train"], train_tfms),
        "val": LC25000Dataset(manifest[manifest["split"] == "val"], eval_tfms),
        "test": LC25000Dataset(manifest[manifest["split"] == "test"], eval_tfms),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    summary: Dict[str, Dict[str, int] | int] = {
        "num_images": int(len(manifest)),
        "class_distribution": manifest["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }
    return loaders, manifest, summary