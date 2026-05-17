# YOLO-LITE Activation Ablation Study Results

Baseline model corresponds to 'Leaky' (LeakyReLU with alpha=0.1).

| Variant   |   Val Loss |   mAP@0.5 |   CPU FPS |
|:----------|-----------:|----------:|----------:|
| Silu      |    56.555  |         0 |      13.4 |
| Mish      |    54.6196 |         0 |       9.1 |
| Hardswish |    90.5697 |         0 |      13.7 |

*Note: mAP@0.5 and validation loss reflect best-epoch performance.*