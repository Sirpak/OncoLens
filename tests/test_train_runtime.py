import copy
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path

sys.modules.setdefault("yaml", types.ModuleType("yaml"))
hf_module = types.ModuleType("huggingface_hub")
hf_module.HfApi = object
sys.modules.setdefault("huggingface_hub", hf_module)

from src.train import resolve_runtime_paths


class TrainRuntimePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = {
            "project": {"seed": 42},
            "dataset": {"path": "/default/data", "image_size": 224},
            "training": {"output_dir": "/default/output"},
        }

    def test_cli_args_take_precedence_over_sagemaker_env(self) -> None:
        args = Namespace(dataset_path="/cli/data", output_dir="/cli/output")
        env = {
            "SM_CHANNEL_TRAINING": "/sm/data",
            "SM_OUTPUT_DATA_DIR": "/sm/output",
            "SM_MODEL_DIR": "/sm/model",
        }

        config, output_dir, model_dir = resolve_runtime_paths(copy.deepcopy(self.base_config), args, env)

        self.assertEqual(config["dataset"]["path"], "/cli/data")
        self.assertEqual(config["training"]["output_dir"], "/cli/output")
        self.assertEqual(output_dir, Path("/cli/output"))
        self.assertEqual(model_dir, Path("/sm/model"))

    def test_sagemaker_env_is_used_when_cli_args_are_missing(self) -> None:
        args = Namespace(dataset_path=None, output_dir=None)
        env = {
            "SM_CHANNEL_TRAINING": "/sm/data",
            "SM_OUTPUT_DATA_DIR": "/sm/output",
            "SM_MODEL_DIR": "/sm/model",
        }

        config, output_dir, model_dir = resolve_runtime_paths(copy.deepcopy(self.base_config), args, env)

        self.assertEqual(config["dataset"]["path"], "/sm/data")
        self.assertEqual(config["training"]["output_dir"], "/sm/output")
        self.assertEqual(output_dir, Path("/sm/output"))
        self.assertEqual(model_dir, Path("/sm/model"))

    def test_local_defaults_are_preserved_without_sagemaker_env(self) -> None:
        args = Namespace(dataset_path=None, output_dir=None)

        config, output_dir, model_dir = resolve_runtime_paths(copy.deepcopy(self.base_config), args, {})

        self.assertEqual(config["dataset"]["path"], "/default/data")
        self.assertEqual(config["training"]["output_dir"], "/default/output")
        self.assertEqual(output_dir, Path("/default/output"))
        self.assertEqual(model_dir, Path("/default/output"))


if __name__ == "__main__":
    unittest.main()