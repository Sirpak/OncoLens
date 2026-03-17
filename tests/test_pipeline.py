import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data_loader import CLASS_NAMES, build_manifest, create_dataloaders, load_image_table
from src.inference import generate_gradcam, load_checkpoint, predict_image
from src.model import build_model


def make_dummy_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for class_index, label in enumerate(CLASS_NAMES):
            class_dir = self.root / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(5):
                make_dummy_image(
                    class_dir / f"sample_{image_index}.jpg",
                    (class_index * 30, image_index * 20, 100),
                )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_load_image_table_and_manifest_are_stratified(self) -> None:
        frame = load_image_table(self.root)
        self.assertEqual(len(frame), 25)
        self.assertEqual(set(frame["label"]), set(CLASS_NAMES))

        manifest = build_manifest(frame, val_size=0.2, test_size=0.2, seed=42)
        self.assertEqual(len(manifest), 25)
        self.assertEqual(manifest["split"].value_counts().to_dict(), {"train": 15, "val": 5, "test": 5})

        split_label_counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
        for label in CLASS_NAMES:
            self.assertEqual(int(split_label_counts.loc["train", label]), 3)
            self.assertEqual(int(split_label_counts.loc["val", label]), 1)
            self.assertEqual(int(split_label_counts.loc["test", label]), 1)

    def test_create_dataloaders_returns_expected_shapes(self) -> None:
        config = {
            "project": {"seed": 42},
            "dataset": {
                "path": str(self.root),
                "image_size": 32,
                "val_size": 0.2,
                "test_size": 0.2,
                "num_workers": 0,
            },
            "training": {"batch_size": 4},
        }

        loaders, manifest, summary = create_dataloaders(config)
        self.assertEqual(set(loaders.keys()), {"train", "val", "test"})
        self.assertEqual(len(manifest), 25)
        self.assertEqual(summary["num_images"], 25)
        self.assertEqual(summary["split_counts"], {"train": 15, "val": 5, "test": 5})

        images, labels = next(iter(loaders["train"]))
        self.assertEqual(images.shape[1:], (3, 32, 32))
        self.assertLessEqual(images.shape[0], 4)
        self.assertEqual(labels.ndim, 1)

    def test_checkpoint_prediction_and_gradcam(self) -> None:
        config = {"model": {"backbone": "resnet18", "pretrained": False}}
        model, _ = build_model(config, len(CLASS_NAMES))
        checkpoint_path = self.root / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": 32,
                "config": config,
            },
            checkpoint_path,
        )

        bundle = load_checkpoint(checkpoint_path, device="cpu")
        image = Image.new("RGB", (32, 32), color=(120, 80, 160))
        prediction = predict_image(bundle, image)
        gradcam = generate_gradcam(bundle, image)

        self.assertIn(prediction["predicted_class"], CLASS_NAMES)
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertEqual(set(prediction["probabilities"].keys()), set(CLASS_NAMES))
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)
        self.assertEqual(gradcam.size, image.size)


if __name__ == "__main__":
    unittest.main()