# yololite — PyTorch reimplementation of YOLO-LITE
# Activation function ablation study package
#
# Original YOLO-LITE paper: "YOLO-LITE: A Real-Time Object Detection Algorithm
# Optimized for Non-GPU Computers" (Huang et al., 2018)
#
# This package provides a configurable PyTorch version of YOLO-LITE that
# supports swapping activation functions (ReLU, LeakyReLU, SiLU, Mish,
# HardSwish) to study their effect on detection performance and speed.

__version__ = "1.0.0"
__author__ = "YOLO-LITE Activation Study"

from yololite.activations import get_activation, ACTIVATION_REGISTRY
from yololite.config import YOLOLiteConfig, load_config
from yololite.model import YOLOLite

__all__ = [
    "get_activation",
    "ACTIVATION_REGISTRY",
    "YOLOLiteConfig",
    "load_config",
    "YOLOLite",
]
