# OncoLens

OncoLens is a clean, Colab-first histopathology classification project for the **LC25000** dataset. Training is designed to run on **Google Colab GPU**, while predictions and experiment outputs are explored locally through a **Streamlit dashboard**.

## What this repo contains

- **Colab notebook** for GPU training
- **PyTorch training pipeline** with configurable dataset paths
- **Inference utilities** for loading checkpoints and running predictions
- **Streamlit dashboard** for predictions, probability charts, confusion matrix, class distribution, metrics, and sample image browsing

> The LC25000 image dataset is **not stored in this repository**. Keep it in Google Drive or another external location and point the config or CLI arguments to that path.

## LC25000 dataset

The project targets the **Lung and Colon Cancer Histopathological Image Dataset (LC25000)**, a 25,000-image dataset spanning five classes:

- `colon_aca` — colon adenocarcinoma
- `colon_n` — benign colon tissue
- `lung_aca` — lung adenocarcinoma
- `lung_n` — benign lung tissue
- `lung_scc` — lung squamous cell carcinoma

The training code expects images to live in class folders somewhere under a configurable dataset root, such as:

- `/content/drive/MyDrive/LC25000/lung_colon_image_set`
- `C:/Users/<you>/Desktop/LC25000/lung_colon_image_set`

## Architecture

- `notebooks/train_on_colab.ipynb` — Colab workflow for GPU training
- `src/data_loader.py` — dataset discovery, transforms, manifest creation, dataloaders
- `src/model.py` — model factory for transfer learning backbones
- `src/train.py` — training loop, metrics export, checkpoint saving, optional Hugging Face Hub upload
- `src/inference.py` — checkpoint loading, prediction utilities, Grad-CAM generation
- `dashboard/streamlit_app.py` — browser UI for model inspection and inference
- `config/config.yaml` — project, dataset, training, storage, and dashboard settings

## ML workflow

1. Store LC25000 outside the repo.
2. Mount Google Drive in Colab.
3. Point `dataset.path` to the external dataset root.
4. Run `src/train.py` on Colab GPU.
5. Save `best_model.pt` and `metrics.json` to Google Drive or optionally upload them to Hugging Face Hub.
6. Run the Streamlit dashboard locally and load those artifacts for visualization.

Tracked outputs include:

- epoch-by-epoch train/validation loss
- epoch-by-epoch train/validation accuracy
- confusion matrix
- classification report
- class distribution summary
- best checkpoint metadata

## Train in Google Colab

Open `notebooks/train_on_colab.ipynb` in Colab and enable **GPU** from `Runtime -> Change runtime type`.

Typical workflow:

1. Clone the repo in Colab.
2. Install `requirements.txt`.
3. Mount Google Drive.
4. Set:
   - dataset path, e.g. `/content/drive/MyDrive/LC25000/lung_colon_image_set`
   - output path, e.g. `/content/drive/MyDrive/OncoLens/runs/exp1`
5. Run:
   - `python src/train.py --config config/config.yaml --dataset-path <DATASET_PATH> --output-dir <OUTPUT_DIR>`

Optional Hugging Face Hub upload:

- set `storage.hf_repo_id` in `config/config.yaml`
- add your token in Colab: `os.environ["HF_TOKEN"] = "..."`

## Run the dashboard locally

Install dependencies:

- `pip install -r requirements.txt`

Start Streamlit from the repository root:

- `streamlit run dashboard/streamlit_app.py`

In the dashboard you can:

- load a trained checkpoint
- load `metrics.json`
- upload a pathology image for prediction
- inspect class probabilities
- view Grad-CAM overlays
- view training curves
- inspect the confusion matrix
- browse example histopathology images from an external folder

## Dataset and artifact safety

The repo is configured to never push common dataset and artifact paths. `.gitignore` excludes:

- `datasets/`, `data/`, `images/`
- `models/`, `artifacts/`, `checkpoints/`, `outputs/`
- `*.pth`, `*.pt`, `*.h5`, `*.ckpt`, `*.csv`
- `.env`, `*.env`

This keeps the GitHub repository lightweight and prevents large images, checkpoints, and environment secrets from being committed.

## Notes for future extensions

- Grad-CAM is already scaffolded for dashboard visualization.
- The saved metrics support confusion matrix and class distribution charts.
- The structure is ready for future expansion into more advanced experiment tracking or cloud training workflows if needed.