# Hyperspectral Imaging for Quality Assessment of Processed Foods: A Case Study on Sugar Content in Apple Jam
<img width="4271" height="2484" alt="Picture1" src="https://github.com/user-attachments/assets/524db8f0-99a2-414a-85c5-cc3b3d959f6a" />

This repository accompanies our study on **non-destructive sugar content estimation** in apple jam using **VNIR hyperspectral imaging (HSI)** and machine learning. It includes a reproducible set of Jupyter notebooks covering preprocessing, dataset construction, and model training/evaluation with classical ML and deep learning.


---

## Dataset
The Apples_HSI dataset is available on Hugging Face:
[issai/Apples_HSI](https://huggingface.co/datasets/issai/Apples_HSI).

### Dataset structure

```text
Apples_HSI/
├── Catalogs/                                    # per-cultivar & sugar-ratio sessions
│   ├── apple_jam_{cultivar}_{sugar proportion}_{apple proportion}_{date}/   # e.g., apple_jam_gala_50_50_17_Dec
│   │   ├── {sample_id}/                         # numeric sample folders (e.g., 911, 912, …)
│   │   │   ├── capture/                         # raw camera outputs + references
│   │   │   │   ├── {sample_id}.raw              # raw hyperspectral cube
│   │   │   │   ├── {sample_id}.hdr              # header/metadata for the raw cube
│   │   │   │   ├── DARKREF_{sample_id}.raw      # dark reference (raw)
│   │   │   │   ├── DARKREF_{sample_id}.hdr
│   │   │   │   ├── WHITEREF_{sample_id}.raw     # white reference (raw)
│   │   │   │   └── WHITEREF_{sample_id}.hdr
│   │   │   ├── metadata/
│   │   │   │   └── {sample_id}.xml              # per-sample metadata/annotations
│   │   │   ├── results/                         # calibrated reflectance + previews
│   │   │   │   ├── REFLECTANCE_{sample_id}.dat  # ENVI-style reflectance cube
│   │   │   │   ├── REFLECTANCE_{sample_id}.hdr
│   │   │   │   ├── REFLECTANCE_{sample_id}.png  # reflectance preview
│   │   │   │   ├── RGBSCENE_{sample_id}.png     # RGB scene snapshot
│   │   │   │   ├── RGBVIEWFINDER_{sample_id}.png
│   │   │   │   └── RGBBACKGROUND_{sample_id}.png
│   │   │   ├── manifest.xml                     # per-sample manifest
│   │   │   ├── {sample_id}.png                  # sample preview image
│   │   │   └── .validated                       # empty marker file
│   │   └── …                                    # more samples
│   └── …                                        # more cultivar/ratio/date folders
│
├── .cache/                                      # service files (upload tool)
├── ._.cache
├── ._paths.rtf
├── .gitattributes                               # LFS rules for large files
└── paths.rtf                                    # path list (RTF)
```

## Repository structure
This repository contains:
- **Pre-processing**: `1_preprocessing.ipynb` (import HSI, calibration, masking (SAM), ROI crop, grid subdivision).
- **Dataset building**: `2_dataset preparation.ipynb` (train/val/test splits, sugar concentration/apple cultivar splits, average spectral vectors extraction).
- **Model training & evaluation**:
  - `3_svm.ipynb` — SVM, scaling, hyperparameter search.
  - `4_xgboost.ipynb` — XGBoost, tuning & early stopping.
  - `5_resnet.ipynb` — 1D ResNet training loops, checkpoints, metrics.



## Preprocessing → Dataset → Models (How to Run)

### 1) **Preprocessing**  

Inputs to set (near the bottom of the notebook)
```python
input_root = "path/to/input"     # root that contains the dataset folders (e.g., Apples_HSI/Catalogs)
output_root = "path/to/output"   # where the NPZ files will be written
paths_txt = "path/to/paths.txt"  # text file with relative paths to .hdr files (one per line)
```
- Run all cells. The notebook:
  - reads `REFLECTANCE_*.hdr` with `spectral.open_image`
  - builds a SAM mask (ref pixel `(255, 247)`, threshold `0.19`)
  - crops ROI and saves `cropped_{ID}.npz` under `output_root/...`

- Each NPZ contains: `cube` (cropped H×W×Bands), `offset` (`y_min`, `x_min`), `metadata` (JSON).


### 2) **Dataset building**  

Run all cells. The notebook:
- loads each NPZ (`np.load(path)["cube"]`)
- extracts **mean spectra per patch** for grid sizes **1, ..., 5**
- creates tables with columns `band_0..band_(B-1)`, `apple_content`, `apple_type`
- writes splits per grid:
  - **apple-based:** `{g}x{g}_train_apple.csv`, `{g}x{g}_val_apple.csv`, `{g}x{g}_test_apple.csv`
  - **rule-based:** `{g}x{g}_train_rule.csv`, `{g}x{g}_val_rule.csv`, `{g}x{g}_test_rule.csv`


### 3) **Model training**  

Classical ML — `3_svm.ipynb`
Run all cells. The notebook:
- loads pre-split CSVs (e.g., `{g}x{g}_train_apple.csv`, `{g}x{g}_test_apple.csv`)
- scales inputs and targets with **MinMaxScaler**
- fits **SVR** with hyperparameters: `C=110`, `epsilon=0.2`, `gamma="scale"`
- reports **RMSE / MAE / R²** on Train/Test (targets inverse-transformed)

Classical ML — `4_xgboost.ipynb`
Run all cells. The notebook:
- loads Train/Val/Test CSVs and scales inputs with **MinMaxScaler**
- builds **DMatrix** and trains with:
    objective = "reg:squarederror", eval_metric = "rmse",
    max_depth = 2, eta = 0.15, subsample = 0.8, colsample_bytree = 1.0,
    lambda = 2.0, alpha = 0.1, seed = 42
    num_boost_round = 400, early_stopping_rounds = 40
- evaluates and prints **RMSE / MAE / R²** (Train/Test)

Deep model — `5_resnet.ipynb`
Run all cells. The notebook:
- builds a **ResNet1D** and DataLoaders (`batch_size=16`)
- trains with **Adam** (`lr=1e-3`, `weight_decay=1e-4`), **epochs=150**, **MAE** loss
- uses target **MinMaxScaler** (inverse-transforms predictions for metrics)
- early-stopping on **Val MAE**; saves best checkpoint to **`best_resnet1d_model.pth`**
- reports **RMSE / MAE / R²** on the Test set


## Downloading the Repository

```bash
git clone <THIS_REPO_URL>
cd <THIS_REPO_FOLDER>
```

## If you use the dataset/source code/pre-trained models in your research, please cite our work:

