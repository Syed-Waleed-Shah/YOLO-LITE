# Modernizing YOLO-LITE: An Ablation Study on Activation Functions for Lightweight Object Detection

## 1. Abstract
YOLO-LITE is an extremely lightweight object detection architecture originally designed to achieve real-time inference (~10 FPS) on non-GPU computers without requiring deep networks or complex attention mechanisms. However, the original architecture relies on the LeakyReLU activation function. In this study, we investigate whether substituting LeakyReLU with modern, smooth, non-monotonic activation functions—specifically SiLU (Swish), Mish, and HardSwish—can improve Mean Average Precision (mAP) without significantly compromising the strict FPS constraints necessary for edge deployment. 

## 2. Objective
The primary objective of this experiment is to identify the optimal activation function for YOLO-LITE that balances computational speed on low-end hardware with feature expressivity. We aim to determine if modern activations can solve the gradient propagation issues inherent in shallow networks (8 convolutional layers).

## 3. Methodology

### 3.1 Model Architecture
We reconstruct the original YOLO-LITE `trial13` architecture in PyTorch:
- **Input:** $224 \times 224 \times 3$
- **Backbone:** 7 Convolutional blocks (Conv2d $\to$ BatchNorm2d $\to$ Activation) interleaved with Max Pooling.
- **Detection Head:** 1 Convolutional layer outputting a $7 \times 7$ grid with 5 anchors.

### 3.2 Activation Functions Analyzed
We evaluate the following activation functions:

1. **ReLU**: $f(x) = \max(0, x)$
   - *Theory*: Fast, sparse, but suffers from the "dying ReLU" problem.
2. **LeakyReLU (Baseline)**: $f(x) = \max(0.1x, x)$
   - *Theory*: Solves dying ReLU; standard in YOLOv2.
3. **SiLU (Swish)**: $f(x) = x \cdot \sigma(x)$
   - *Theory*: Smooth and non-monotonic. Helps gradient flow in deep networks, but requires computing an exponential function.
4. **Mish**: $f(x) = x \cdot \tanh(\ln(1 + e^x))$
   - *Theory*: Bounded below, unbounded above. Often yields better performance than SiLU due to smoother gradients, but is computationally heavier.
5. **HardSwish**: $f(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}$
   - *Theory*: A piecewise linear approximation of SiLU. Designed for mobile devices as it relies only on basic arithmetic (add, multiply, max).

### 3.3 Experimental Setup
- **Dataset**: Pascal VOC (2007 + 2012)
- **Optimizer**: SGD (LR=0.001, Momentum=0.9, Weight Decay=5e-4)
- **Batch Size**: 8 (with Gradient Accumulation = 4, effective batch size 32)
- **Hardware**: CPU / Low-end GPU (Mixed Precision enabled)

---

## 4. Results

### 4.1 Ablation Study Table
*(Note: Run `python ablation_study.py` to populate this table from logs)*

| Variant | Val Loss | mAP@0.5 | CPU FPS | GPU FPS | Model Size (MB) |
|---------|----------|---------|---------|---------|-----------------|
| Leaky   | -        | -       | -       | -       | ~62.0           |
| ReLU    | -        | -       | -       | -       | ~62.0           |
| SiLU    | -        | -       | -       | -       | ~62.0           |
| Mish    | -        | -       | -       | -       | ~62.0           |
| HardSwish| -       | -       | -       | -       | ~62.0           |

### 4.2 Visualizations
*(Embed generated graphs from `logs/plots/` here)*

![Validation Loss Curves](../logs/plots/val_loss_curves.png)
![CPU FPS Comparison](../logs/plots/cpu_fps_comparison.png)

---

## 5. Comparison Analysis

**mAP vs FPS Trade-off:**
- *Observations will be written here after executing the training loops.*
- Example: "While Mish achieved the highest mAP, the computational overhead of calculating `tanh` and `softplus` reduced the CPU FPS by X%, making it unsuitable for the original YOLO-LITE non-GPU constraint."
- Example: "HardSwish provided near-SiLU accuracy while recovering the FPS lost to exponential calculations, making it the most viable modern replacement."

## 6. Conclusion
*(Write conclusion based on the generated data. Identify the definitive best activation function for Edge-AI deployment of YOLO-LITE).*

## 7. References
1. Huang, R., Pedoeem, J., & Chen, C. (2018). YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers. *arXiv:1811.05588*.
2. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation Functions. *arXiv:1710.05941*.
3. Misra, D. (2019). Mish: A Self Regularized Non-Monotonic Activation Function. *arXiv:1908.08681*.
4. Howard, A., et al. (2019). Searching for MobileNetV3. *arXiv:1905.02244*.
