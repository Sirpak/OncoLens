from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from tqdm import tqdm

from src.data_loader import CLASS_NAMES, create_dataloaders
from src.model import build_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); total_loss = total_correct = total_seen = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); logits = model(images); loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item(); total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, loader, criterion, device):
    model.eval(); total_loss = total_correct = total_seen = 0; y_true = []; y_pred = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images); loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item(); total_seen += labels.size(0)
            y_true.extend(labels.cpu().tolist()); y_pred.extend(preds.cpu().tolist())
    return total_loss / total_seen, total_correct / total_seen, y_true, y_pred
def resolve_runtime_paths(config: dict, args: argparse.Namespace, environ: dict | None = None) -> tuple[dict, Path, Path]:
    runtime_env = environ or os.environ
    sm_training_dir = runtime_env.get("SM_CHANNEL_TRAINING")
    sm_output_dir = runtime_env.get("SM_OUTPUT_DATA_DIR")
    sm_model_dir = runtime_env.get("SM_MODEL_DIR")

    if args.dataset_path:
        config["dataset"]["path"] = args.dataset_path
    elif sm_training_dir:
        config["dataset"]["path"] = sm_training_dir

    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    elif sm_output_dir:
        config["training"]["output_dir"] = sm_output_dir

    output_dir = Path(config["training"]["output_dir"]).expanduser()
    model_dir = Path(sm_model_dir).expanduser() if sm_model_dir else output_dir
    return config, output_dir, model_dir


def maybe_upload_to_hub(output_dir: Path, model_dir: Path, config: dict) -> None:
    repo_id = config.get("storage", {}).get("hf_repo_id", "").strip()
    if not repo_id:
        return
    token = os.getenv(config.get("storage", {}).get("hf_token_env", "HF_TOKEN"))
    if not token:
        print("HF upload skipped: token not found."); return
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    artifact_locations = {
        "best_model.pt": model_dir,
        "metrics.json": output_dir,
        "config.yaml": output_dir,
    }
    for name, directory in artifact_locations.items():
        path = directory / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id, repo_type="model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config, output_dir, model_dir = resolve_runtime_paths(config, args)

    set_seed(int(config["project"]["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    loaders, manifest, summary = create_dataloaders(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(config, len(CLASS_NAMES)); model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy = -1.0; best_truth = []; best_pred = []
    for epoch in range(int(config["training"]["epochs"])):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss); history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch + 1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_accuracy:
            best_accuracy = val_acc; best_truth = y_true; best_pred = y_pred
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": int(config["dataset"]["image_size"]),
                    "config": config,
                },
                model_dir / "best_model.pt",
            )

    metrics = {
        "history": history,
        "best_val_accuracy": best_accuracy,
        "class_names": CLASS_NAMES,
        "class_distribution": summary["class_distribution"],
        "split_counts": summary["split_counts"],
        "num_images": summary["num_images"],
        "manifest_rows": int(len(manifest)),
        "confusion_matrix": confusion_matrix(best_truth, best_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "classification_report": classification_report(
            best_truth, best_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    maybe_upload_to_hub(output_dir, model_dir, config)


if __name__ == "__main__":
    main()