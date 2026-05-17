"""
config.py — Experiment Configuration System
============================================

Provides a typed dataclass configuration that can be loaded from:
  1. YAML file          : load_config("experiments/configs/silu.yaml")
  2. Command-line args  : parse_args() → returns YOLOLiteConfig
  3. Direct Python      : YOLOLiteConfig(activation="mish")

All values have sensible defaults for low-end hardware (CPU / 2 GB VRAM GPU).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# YAML is optional — graceful fallback to manual parsing
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class YOLOLiteConfig:
    """
    Master configuration for a YOLO-LITE training / benchmarking run.

    Activation selection
    --------------------
    activation : str
        One of: 'relu', 'leaky', 'silu', 'mish', 'hardswish'
        Default is 'leaky' to match the original YOLO-LITE paper.

    Hardware / memory optimisations
    --------------------------------
    batch_size      : small default (8) for low-VRAM machines
    grad_accum      : simulate larger effective batch without extra VRAM
    amp             : automatic mixed precision (FP16 compute, FP32 master)
    num_workers     : 0 on Windows (avoids multiprocessing pickling issues)
    """

    # --- Model ---
    activation: str = "leaky"           # Activation function name
    num_classes: int = 20               # VOC=20, COCO=80
    input_size: int = 224               # Input resolution (224 for YOLO-LITE)
    num_anchors: int = 5

    # --- Training ---
    epochs: int = 10
    batch_size: int = 8                 # Low-VRAM default
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    burn_in: int = 1000                 # LR warmup steps
    lr_steps: list = field(default_factory=lambda: [40000, 45000])
    lr_scales: list = field(default_factory=lambda: [0.1, 0.1])

    # --- Low-resource optimisations ---
    grad_accum: int = 1                 # Gradient accumulation steps
    amp: bool = False                   # Mixed precision training
    num_workers: int = 0               # DataLoader workers (0 = main process)
    pin_memory: bool = False

    # --- Dataset ---
    dataset: str = "voc"               # 'voc' or 'coco'
    data_root: str = "./data"          # Root dir for dataset download
    voc_years: list = field(default_factory=lambda: ["2007", "2012"])

    # --- Checkpointing & logging ---
    run_name: str = "experiment"       # Used as CSV log filename prefix
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10               # Save checkpoint every N epochs
    resume: Optional[str] = None       # Path to .pt checkpoint to resume from

    # --- Inference / benchmarking ---
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    warmup_iters: int = 10             # Warmup before FPS measurement

    # --- YOLO loss weights (match original Darknet defaults) ---
    object_scale: float = 5.0
    noobject_scale: float = 1.0
    class_scale: float = 1.0
    coord_scale: float = 1.0

    def __post_init__(self):
        # Derive run_name from activation if not explicitly set
        if self.run_name == "experiment":
            self.run_name = f"{self.activation}_run"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def csv_log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.run_name}.csv")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}_best.pt")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML Loader
# ---------------------------------------------------------------------------

def load_config(yaml_path: str, overrides: Optional[dict] = None) -> YOLOLiteConfig:
    """
    Load a YOLOLiteConfig from a YAML file with optional key overrides.

    Parameters
    ----------
    yaml_path : str
        Path to a YAML config file (see experiments/configs/ for examples).
    overrides : dict, optional
        Key-value pairs that override YAML values (e.g. from argparse).

    Returns
    -------
    YOLOLiteConfig
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. "
            "Install it with: pip install pyyaml"
        )

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    # Filter only known fields
    known = {f.name for f in YOLOLiteConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}

    return YOLOLiteConfig(**filtered)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """
    Build a reusable argparse ArgumentParser that maps to YOLOLiteConfig fields.
    Can be extended by train.py, benchmark.py, inference.py, etc.
    """
    p = argparse.ArgumentParser(
        description="YOLO-LITE Activation Study — configurable training & inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Quick switches ---
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides defaults; CLI args override YAML)"
    )
    p.add_argument(
        "--activation", type=str, default=None,
        choices=["relu", "leaky", "silu", "mish", "hardswish"],
        help="Activation function to use in all conv layers"
    )
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--input_size",  type=int, default=None)

    # --- Training ---
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--learning_rate",  type=float, default=None)
    p.add_argument("--weight_decay",   type=float, default=None)
    p.add_argument("--grad_accum",     type=int,   default=None,
                   help="Gradient accumulation steps (simulates larger batch)")

    # --- Low-resource flags ---
    p.add_argument("--amp", action="store_true", default=None,
                   help="Enable automatic mixed precision (requires CUDA)")
    p.add_argument("--num_workers", type=int, default=None)

    # --- Dataset ---
    p.add_argument("--dataset",   type=str, default=None, choices=["voc", "coco"])
    p.add_argument("--data_root", type=str, default=None)

    # --- Logging ---
    p.add_argument("--run_name",        type=str, default=None)
    p.add_argument("--log_dir",         type=str, default=None)
    p.add_argument("--checkpoint_dir",  type=str, default=None)
    p.add_argument("--resume",          type=str, default=None)

    # --- Misc ---
    p.add_argument("--dry_run", action="store_true",
                   help="Run one batch for debugging, then exit")
    p.add_argument("--seed", type=int, default=42)

    return p


def parse_args(extra_args: Optional[list] = None) -> tuple[YOLOLiteConfig, argparse.Namespace]:
    """
    Parse CLI arguments and return (config, namespace).
    If --config is given, loads YAML first then applies CLI overrides.

    Usage
    -----
    cfg, args = parse_args()
    print(cfg.activation)  # 'silu'
    """
    parser = build_argparser()
    args = parser.parse_args(extra_args)

    # Collect non-None CLI overrides
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ("config", "dry_run", "seed")
    }

    if args.config:
        cfg = load_config(args.config, overrides=cli_overrides)
    else:
        # Start from defaults, apply overrides
        defaults = asdict(YOLOLiteConfig())
        defaults.update(cli_overrides)
        cfg = YOLOLiteConfig(**{
            k: v for k, v in defaults.items()
            if k in YOLOLiteConfig.__dataclass_fields__
        })

    return cfg, args
