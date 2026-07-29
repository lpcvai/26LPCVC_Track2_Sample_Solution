# LPCVC-Track-2 :movie_camera: :zap:

This repository contains the solution for LPCVC 2026 Track 2: Video Classification with Dynamic Frame Selection. Our approach modifies PyTorch Vision's video classification pipeline to handle the QEVD dataset with optimized frame sampling.

## :fire: Overview

- Modified `pytorch/vision` for dynamic frame selection
- Implemented dataset preprocessing utilities for QEVD
- Added video validation and corruption checking tools
- Trained sample solution

---

## :rocket: Sample Solution

Try out the sample solution consisting of our most recent model checkpoint [here](https://drive.google.com/file/d/1vAJdpMRdJZPOPSkcVDyKu2MXOXb8qyiS/view?usp=drive_link). Read about training and evaluating the solution in the steps below.

---

## 0. Environment Setup :wrench:

### :point_right: Prerequisites

Make sure you have Python 3.10 or higher installed on your system. This repository was tested with 3.10.

### Optional (recommended): Create a conda Environment

You can use a conda environment to avoid dependency conflicts for the Track 2 repository. Create and activate one with the following:

Please ensure that you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) downloaded.

To verify installation, open Anaconda Prompt or Terminal through Start Menu and type this:

```bash
conda --version
```

Create a new conda environment by running the following command, naming the environment whatever you'd like:

```bash
conda create --name lpcvc python=3.10 -y
```

Activate the conda environment that you crated by running the following command:

```bash
conda activate lpcvc
```

If you ever need to deactivate from the conda environment, you can run the following command:

```bash
conda deactivate
```

### Install Dependencies

Run this command to install the Python dependencies regardless if you are in an enviroment:

```bash
pip install -r requirements.txt
```

This will install all the packages used for data preprocessing, model training, and video processing.

---

## 1. Modified Files :memo:

We modified the following files from the `pytorch/vision` repository:

| File                                       | Description                                |
| ------------------------------------------ | ------------------------------------------ |
| `references/video_classification/train.py` | Training script with custom configurations |
| `torchvision/datasets/video_utils.py`      | Dynamic frame selection implementation     |

---

## 2. Dataset Preprocessing :open_file_folder:

### :file_folder: Downloading the QEVD Dataset

To download the QEVD dataset, please refer to the instructions for Qualcomm's [QEVD dataset](https://www.qualcomm.com/developer/software/qevd-dataset) link.

### :point_right: Refactoring the QEVD Dataset

We use `refactor_dataset.py` to organize the QEVD dataset into the required directory structure.

```python
# Example usage in refactor_dataset.py
srcs = []
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-1'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-2'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-3'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-4'))
dest = Path('./QEVD_organized')

refactor = DatasetRefactorer(srcs, dest, Path('./dataset/fine_grained_labels_release.json'))
refactor.refactor_dataset()
```

The script organizes videos into the following structure:

```text
root/
├── train/
│   ├── action_category_1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── action_category_2/
└── val/
    ├── action_category_1/
    └── action_category_2/
```

### :broom: Cleaning Corrupted Videos

We use `check_videos.py` to validate and clean corrupted videos from the dataset.

```bash
# Check and quarantine corrupted videos
python check_videos.py --root ./full_dataset \
    --quarantine ./bad_videos \
    --remux \
    --replace \
    --ext mp4 \
    --jobs 8 \
    --report bad_videos.csv
```

---

## 3. Training the Model :weight_lifting:

### :point_right: Training Command

Run the training script with the following arguments:

```bash
python references/video_classification/train.py \
    --data-path /path/to/dataset/root \
    --weights KINETICS400_V1 \
    --cache-dataset \
    --epochs 15 \
    --batch-size 24 \
    --lr 0.01
```

### :gear: Key Parameters

| Parameter         | Description                                                    | Example            |
| ----------------- | -------------------------------------------------------------- | ------------------ |
| `--data-path`     | Path to dataset root (`root/train` or `val/action_categories`) | `./full_dataset/`  |
| `--resume`        | Path to checkpoint for resuming training                       | `./checkpoint.pth` |
| `--start-epoch`   | Starting epoch when resuming from checkpoint                   | `10`               |
| `--weights`       | Pre-trained weights to use                                     | `KINETICS400_V1`   |
| `--cache-dataset` | Cache processed dataset for faster loading                     | (flag)             |
| `--batch-size`    | Batch size per GPU                                             | `24`               |
| `--epochs`        | Total number of epochs                                         | `15`               |
| `--lr`            | Initial learning rate                                          | `0.01`             |

### :warning: Important Notes

- For distributed training, it's recommended to pre-compute the dataset cache on a single GPU first
- The model uses `r2plus1d_18` architecture by default
- Layer freezing is enabled for faster training (only `layer4` and `fc` are trainable)

**More details on parameters can be found in the `get_args_parser()` function in `train.py`**

---

## 4. Model Validation :white_check_mark:

### :point_right: Validation Command

To validate a trained model, run:

```bash
python references/video_classification/train.py \
    --data-path /path/to/dataset/root \
    --resume /path/to/checkpoint.pth \
    --test-only
```

### :bar_chart: Evaluation Metrics

The validation process reports:

- **Clip Accuracy (Acc@1, Acc@5)**: Accuracy per video clip
- **Video Accuracy (Acc@1, Acc@5)**: Aggregated accuracy across all clips of a video

```python
# Aggregation of clip predictions for video-level accuracy
preds = torch.softmax(output, dim=1)
for b in range(video.size(0)):
    idx = video_idx[b].item()
    agg_preds[idx] += preds[b].detach()
    agg_targets[idx] = target[b].detach().item()
```

---

## 5. AI Hub Deployment :rocket:

This section walks through the full pipeline for deploying your trained model to Qualcomm AI Hub — from preprocessing raw videos all the way through on-device inference and accuracy evaluation.

### :electric_plug: Prerequisites

1. **Download the model checkpoint** from the [Sample Solution](#rocket-sample-solution) link above (or use your own trained `.pth` file).

---

### 5a. Data Preprocessing :floppy_disk:

> **Skip this step if you already have preprocessed `.npy` tensors saved.**

If you are starting from raw `.mp4` video files, use `preprocess_and_save.py` to decode, resize, and centre-crop every video into a NumPy tensor that can be fed directly to the model.

**Step 1 — Configure the script.** Open `preprocess_and_save.py` and update the variables at the top:

```python
# Root directory containing class-labelled video folders:
# DATA_ROOT/<class_name>/<video>.mp4
DATA_ROOT = "/path/to/your/videos"

# Where to write the output .npy tensors and manifest.jsonl
OUT_ROOT  = "/path/to/preprocessed_tensors"

# Frame sampling settings — must match the model's expected input
CLIP_LEN   = 16   # frames per clip
FRAME_RATE = 4    # fps used when decoding
```

**Step 2 — Run the script:**

```bash
python preprocess_and_save.py
```

The script will:
- Walk every `.mp4` file under `DATA_ROOT`, preserving the `<class_name>/<video>` hierarchy.
- Decode each video at `FRAME_RATE` fps and extract a `CLIP_LEN`-frame clip.
- Apply the standard `r2plus1d_18` spatial preprocessing (resize to 128×171, centre-crop to 112×112). **Note:** mean/std normalisation is intentionally omitted — it is baked into the exported model.
- Save each clip as a float32 `.npy` tensor of shape `(1, 3, T, 112, 112)` under `OUT_ROOT`.
- Write a `manifest.jsonl` to `OUT_ROOT` that records the video path, class label, tensor path, shape, and dtype for every sample — this file is required by `evaluate.py`.

**Expected output layout:**

```text
OUT_ROOT/
├── manifest.jsonl
├── class_a/
│   ├── video1.npy
│   └── video2.npy
└── class_b/
    └── video3.npy
```

> :warning: The script expects `DATA_ROOT/<class_name>/<video>.mp4`. The class folder name is used as the label in the manifest.

---

### 5b. Exporting and Downloading the Compiled Model :arrow_down:

`example_export.py` is the **primary and recommended** tool for compiling your model on Qualcomm AI Hub and downloading the resulting `.bin` (QNN Context Binary) to your local machine. It handles model tracing, ONNX conversion, AI Hub submission, profiling, and download all in one go.

**Step 1 — Configure the script.** Open `example_export.py` and update the marked sections:

```python
# ① Set the number of output classes to match your training run
num_classes = 92  # e.g. 92 for QEVD

# ② Point to your checkpoint
if os.path.exists("./model.pth"):
    ckpt = torch.load("./model.pth", ...)

# ③ (Optional) Point to your preprocessed tensors for a real inference sanity-check
#    inside custom_sample_inputs(), set:
data_dir = "/path/to/preprocessed_tensors"
#    Leave empty ("") to use a random tensor as a fallback.

# ④ Set the frame count that matches your checkpoint
additional_model_kwargs.setdefault("num_frames", 16)
```

**Step 2 — Run the export:**

```bash
# Compile for the Dragonwing IQ-9075 EVK (default target device)
python example_export.py

# Or specify a different device:
python example_export.py --device "Samsung Galaxy S25"

# To see all available options:
python example_export.py --help
```

By default, the script will:
1. Trace the patched PyTorch model to TorchScript.
2. Submit a **compile job** to AI Hub targeting `QNN_CONTEXT_BINARY`.
3. Download the compiled `.bin` model to `./export_assets/` when the job completes.

Profiling and on-Hub inferencing are **skipped by default** (`skip_profiling=True`, `skip_inferencing=True`) to save time. If you want to also profile during export, pass `--skip-profiling false`.

**Useful flags:**

| Flag | Description | Default |
|---|---|---|
| `--device` | Target device name (run `hub.get_devices()` to list) | `Dragonwing IQ-9075 EVK` |
| `--target-runtime` | Runtime format (`qnn_context_binary`, `tflite`, `onnx`, etc.) | `qnn_context_binary` |
| `--skip-profiling` | Skip device profiling | `true` |
| `--skip-inferencing` | Skip on-Hub inference | `true` |
| `--skip-downloading` | Skip downloading the compiled model | `false` |
| `--output-dir` | Directory to save downloaded model | `./export_assets` |

The downloaded model (`.bin`) will be under `./export_assets/` and is required for `run_inference.py`.

> :bulb: **Why use `example_export.py` over a manual ONNX export?** The `example_export.py` pipeline uses the official `qai_hub_models` compilation path, which applies QNN-level graph optimizations and correctly handles normalization layers. In testing, this yielded significantly better on-device accuracy than a manual `torch.onnx.export` → `compile_and_profile.py` workflow.

---

### 5c. Compile & Profile *(Optional)* :chart_with_upwards_trend:

> :information_source: **This step is optional.** `example_export.py` already compiles (and optionally profiles) the model for you. Use `compile_and_profile.py` only if you already have a standalone ONNX file that you want to compile and profile independently.

`compile_and_profile.py` accepts a pre-exported ONNX model, submits it to AI Hub for compilation, then immediately submits a profiling job with the resulting binary.

**Configure the script:**

```python
ONNX_DIR        = "/path/to/onnx_directory"  # directory containing model.onnx
VIDEO_ONNX_NAME = "model.onnx"
DEVICE_NAME     = "Dragonwing IQ-9075 EVK"

# Input dimensions — must match the model
BATCH, C, T, H, W = 1, 3, 16, 112, 112
```

**Run:**

```bash
python compile_and_profile.py
```

This will:
1. Load and validate the ONNX model.
2. Submit a compile job to AI Hub (`--target_runtime qnn_context_binary`).
3. Wait for compilation to finish, then immediately submit a profile job.

---

### 5d. On-Device Inference :zap:

Once you have either a compiled `.bin` from `example_export.py` **or** an ONNX file, use `run_inference.py` to run the full dataset through the model on AI Hub and collect the output logits.

**Configure the script:**

```python
# --- Model source: choose one ---
# Option A (recommended): pre-compiled binary from example_export.py
DLC_PATH  = "./export_assets/resnet_2plus1d.bin"
ONNX_DIR  = ""   # leave empty when using DLC_PATH

# Option B: compile from ONNX on-the-fly
DLC_PATH  = ""   # leave empty
ONNX_DIR  = "/path/to/onnx_dir"

# --- Data ---
data_path = "/path/to/preprocessed_tensors"   # OUT_ROOT from preprocess_and_save.py
OUTPUT_H5 = "dataset-export.h5"               # where results are written

# --- Input shape (must match your compiled model) ---
BATCH, C, T, H, W = 1, 3, 16, 112, 112

# --- Channel layout ---
# Set True only if the .bin was compiled with channel-last (NTHWC) input.
# The .bin from example_export.py uses channel-first (NCTHW), so keep False.
IS_DLC_CHANNEL_LAST = False

# --- Quick debug mode ---
USE_SINGLE_TENSOR    = False   # True to run only one sample
SINGLE_TENSOR_INDEX  = 0      # which tensor to pick from data_path
```

**Run:**

```bash
python run_inference.py
```

The script:
1. Loads all `.npy` tensors from `data_path`, sorted in the same order as the `manifest.jsonl`.
2. Uploads or compiles the model on AI Hub.
3. Sends the dataset to AI Hub in chunks of 538 samples (to stay under the 2 GB flatbuffer limit).
4. Waits for all inference jobs to complete and collects the output logits.
5. Writes all logits to `OUTPUT_H5` (default `dataset-export.h5`) in HDF5 format for consumption by `evaluate.py`.

> :warning: Make sure `T` (frame count) and `IS_DLC_CHANNEL_LAST` match the model you compiled. Mismatches will cause silent accuracy degradation or shape errors.

---

### 5e. Evaluation :bar_chart:

`evaluate.py` loads the HDF5 output from `run_inference.py`, matches each prediction to the ground-truth label from `manifest.jsonl`, and reports Top-1 and Top-5 accuracy.

**Prerequisites:**
- `dataset-export.h5` — produced by `run_inference.py`
- `manifest.jsonl` — produced by `preprocess_and_save.py` (inside `OUT_ROOT`)
- `class_map.json` — maps class folder names to integer indices (should be in the repo root)

**Run:**

```bash
python evaluate.py \
    --h5 dataset-export.h5 \
    --manifest /path/to/preprocessed_tensors/manifest.jsonl \
    --class_map class_map.json
```

For a quick sanity check on a single sample, first set `USE_SINGLE_TENSOR = True` in `run_inference.py`, run inference, then run `evaluate.py --verbose`.

---

### :arrows_counterclockwise: Full Pipeline Summary

```text
Raw videos
    │
    ▼  preprocess_and_save.py
.npy tensors + manifest.jsonl
    │
    ├──▶  example_export.py  ──▶  compiled .bin (+ optional profile)
    │         ↑ recommended
    └──▶  compile_and_profile.py  (optional, ONNX-only path)
                                           │
                                           ▼
                                 run_inference.py
                                           │
                                           ▼
                                    dataset-export.h5
                                           │
                                           ▼
                                    evaluate.py
                                           │
                                           ▼
                               Top-1 / Top-5 Accuracy
```

---

## :link: References

- Built on [PyTorch Vision](https://github.com/pytorch/vision)
- Dataset: QEVD (Qualcomm Exercise Video Dataset)
- [Qualcomm AI Hub ResNet Model](https://github.com/qualcomm/ai-hub-models/tree/main/qai_hub_models/models/resnet_2plus1d)

---

## Acknowledgements
This material is based upon work supported by the National Science Foundation under Grant Number 2504445. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

---

## Citation

If you use this sample solution or refer to the IEEE Low-Power Computer Vision Challenge, please cite the challenge as follows:

```bibtex
@misc{lpcvc,
  author       = {{IEEE Low Power Computer Vision Challenge Organizing Committee}},
  title        = {{IEEE Low Power Computer Vision Challenge}},
  howpublished = {\url{https://lpcv.ai/}},
  note         = {Annual competition series on low power computer vision}
}
```
