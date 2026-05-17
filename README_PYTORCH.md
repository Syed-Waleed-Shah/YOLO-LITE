# YOLO-LITE PyTorch Reimplementation

This is a PyTorch reimplementation of YOLO-LITE, originally designed for non-GPU computers by Huang et al.
The original repository was based on Darknet (C/C++) and only contained `.cfg` and `.weights` files.
This version introduces a complete Python training pipeline, and specifically focuses on an **Activation Function Ablation Study**.

## Features
- **Standalone PyTorch Implementation**: `yololite` module for model, dataset, loss, and training loop.
- **Activation Factory**: Easily swap between `leaky`, `relu`, `silu`, `mish`, and `hardswish`.
- **Low-VRAM Optimizations**: Support for Mixed Precision (AMP) and Gradient Accumulation, designed to run on limited hardware.
- **Ablation Study Toolkit**: Includes automated benchmarking (`benchmark.py`), visualization (`visualize.py`), and markdown table generation (`ablation_study.py`).

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. To prepare the dataset (Pascal VOC), simply run the training script. It will auto-download via `torchvision` to the `./data` directory (requires ~2GB).

## Running the Ablation Study

To run the full suite of experiments:

```bash
# 1. Run Baseline (LeakyReLU)
python experiments/run_baseline_leaky.py

# 2. Run Variants
python experiments/run_silu.py
python experiments/run_mish.py
python experiments/run_hardswish.py
python experiments/run_baseline_relu.py

# 3. Benchmark FPS and Model Size
python benchmark.py

# 4. Generate Visualizations (saved to logs/plots/)
python visualize.py

# 5. Generate Ablation Markdown Table
python ablation_study.py
```

## Running Inference

You can run inference on a single image using a trained PyTorch checkpoint or original Darknet `.weights` files:

```bash
python inference.py --image test.jpg --weights checkpoints/silu_run_best.pt --activation silu
```

To load original Darknet weights (if available):
```bash
python inference.py --image test.jpg --weights weights/tiny-yolo-voc.weights --activation leaky
```

## Project Structure

- `yololite/`: Core PyTorch module (model architecture, loss, datasets, config, activations).
- `experiments/`: Experiment configurations (`.yaml`) and simple launcher scripts.
- `logs/`: Generated CSV logs, benchmark results, and matplotlib plots.
- `reports/`: Markdown report template for publishing findings.
- `train.py`: Main training loop.
- `benchmark.py`: FPS and parameter counting utilities.
- `visualize.py`: Graph generation for the ablation study.
