"""
model.py — YOLO-LITE Architecture (PyTorch Reimplementation)
=============================================================

Faithful reproduction of the YOLO-LITE architecture from `cfg/tiny-yolov2-trial13.cfg`.
Original paper: "YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers"

Architecture Specs:
- Input: 224x224x3 (RGB)
- 8 Convolutional Layers (16 -> 32 -> 64 -> 128 -> 256 -> 1024 -> 2048 -> 125)
- 6 MaxPooling Layers
- Final output: 7x7 grid, 5 anchors, (5 + num_classes) values per anchor
- For VOC (20 classes): 125 output channels = 5 * (5 + 20)

Changes from original:
- Swappable activation functions via `yololite.activations.get_activation()`
"""

import torch
import torch.nn as nn
from typing import Optional

from yololite.activations import get_activation

class ConvBlock(nn.Module):
    """
    Standard YOLO convolutional block:
    Conv2D -> BatchNorm2D -> Activation
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int, activation_name: str):
        super().__init__()
        
        # YOLO-LITE uses bias=False when using BatchNorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation_name)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOLite(nn.Module):
    """
    YOLO-LITE Object Detection Model.
    Recreates trial13 from the original repository.
    """
    def __init__(self, num_classes: int = 20, activation_name: str = "leaky"):
        """
        Parameters
        ----------
        num_classes : int
            Number of object classes (default 20 for Pascal VOC).
        activation_name : str
            Activation function to use in all conv layers except the last one.
            Options: 'relu', 'leaky', 'silu', 'mish', 'hardswish'.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 5
        self.output_channels = self.num_anchors * (5 + num_classes)
        self.activation_name = activation_name

        # --- Backbone ---
        
        # Layer 1: Conv 16, 3x3, pad 1 + MaxPool 2x2, stride 2
        # Input: 224x224x3 -> Output: 112x112x16
        self.layer1 = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1, activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2: Conv 32, 3x3, pad 1 + MaxPool 2x2, stride 2
        # Input: 112x112x16 -> Output: 56x56x32
        self.layer2 = nn.Sequential(
            ConvBlock(16, 32, 3, 1, 1, activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 3: Conv 64, 3x3, pad 1 + MaxPool 2x2, stride 2
        # Input: 56x56x32 -> Output: 28x28x64
        self.layer3 = nn.Sequential(
            ConvBlock(32, 64, 3, 1, 1, activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 4: Conv 128, 3x3, pad 1 + MaxPool 2x2, stride 2
        # Input: 28x28x64 -> Output: 14x14x128
        self.layer4 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1, activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 5: Conv 256, 3x3, pad 1 + MaxPool 2x2, stride 2
        # Input: 14x14x128 -> Output: 7x7x256
        self.layer5 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1, activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # --- Detection Head ---
        
        # Layer 6: Conv 1024, 3x3, pad 1 + MaxPool 2x2, stride 1, pad 0 -> (output is 6x6 if stride=1, pad=0, wait, darknet maxpool stride=1 does padding to maintain size)
        # Actually, in Darknet: maxpool size=2, stride=1 with pad=0 reduces size by 1.
        # But wait! Let's check `trial13.cfg`:
        # [maxpool] size=2, stride=1. In darknet, if stride=1, it pads to keep same spatial dimension.
        # We use ZeroPad2d( (0,1,0,1) ) to replicate Darknet's maxpool with stride=1
        self.layer6 = nn.Sequential(
            ConvBlock(256, 1024, 3, 1, 1, activation_name),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1) # Output remains 7x7
        )
        
        # Layer 7: Conv 2048, 3x3, pad 1
        # Input: 7x7x1024 -> Output: 7x7x2048
        self.layer7 = ConvBlock(1024, 2048, 3, 1, 1, activation_name)
        
        # Layer 8: Conv 125 (linear activation), 1x1
        # Input: 7x7x2048 -> Output: 7x7x125
        self.layer8 = nn.Conv2d(2048, self.output_channels, kernel_size=1, 
                                stride=1, padding=0, bias=True)
                                
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Returns
        -------
        torch.Tensor
            Shape: (batch_size, num_anchors * (5 + num_classes), 7, 7)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out = self.layer8(x)
        return out

    def _initialize_weights(self):
        """Kaiming normal initialization for convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    # Final layer initialization (helps with initial loss)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_darknet_weights(self, weights_path: str):
        """
        Loads pre-trained Darknet weights (.weights).
        For YOLO-LITE, we skip the 4 header int32s if seen/minor/major, etc.
        """
        import numpy as np
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=4)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for m in self.modules():
            if isinstance(m, ConvBlock):
                conv = m.conv
                bn = m.bn
                
                # Darknet stores: bn_bias, bn_weight, bn_running_mean, bn_running_var, conv_weight
                num_bn_biases = bn.bias.numel()
                
                # Bias
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases]).view_as(bn.bias)
                bn.bias.data.copy_(bn_biases)
                ptr += num_bn_biases
                
                # Weight
                bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases]).view_as(bn.weight)
                bn.weight.data.copy_(bn_weights)
                ptr += num_bn_biases
                
                # Running Mean
                bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases]).view_as(bn.running_mean)
                bn.running_mean.data.copy_(bn_running_mean)
                ptr += num_bn_biases
                
                # Running Var
                bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases]).view_as(bn.running_var)
                bn.running_var.data.copy_(bn_running_var)
                ptr += num_bn_biases
                
                # Conv Weights
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(conv.weight)
                conv.weight.data.copy_(conv_weights)
                ptr += num_weights

            elif isinstance(m, nn.Conv2d) and m.bias is not None:
                # The final layer has a bias and no batch norm
                num_biases = m.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases]).view_as(m.bias)
                m.bias.data.copy_(conv_biases)
                ptr += num_biases
                
                num_weights = m.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(m.weight)
                m.weight.data.copy_(conv_weights)
                ptr += num_weights
                
        print(f"Loaded {ptr} weights from {weights_path}.")

if __name__ == "__main__":
    model = YOLOLite(num_classes=20, activation_name="leaky")
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Expected: [1, 125, 7, 7]
