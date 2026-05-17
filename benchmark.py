"""
benchmark.py — Benchmarking Utility for YOLO-LITE Ablation Study
================================================================

Measures:
1. Model parameter count
2. Model size (MB)
3. Inference FPS (CPU & GPU if available)
"""

import torch
import time
import os
import csv
from yololite.model import YOLOLite
from yololite.config import parse_args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def benchmark_fps(model, device, input_size=224, warmup=10, iters=100):
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    return iters / total_time

def run_benchmark():
    cfg, args = parse_args()
    
    activations = ["relu", "leaky", "silu", "mish", "hardswish"]
    
    os.makedirs(cfg.log_dir, exist_ok=True)
    csv_path = os.path.join(cfg.log_dir, "benchmark_results.csv")
    
    print(f"{'Activation':<15} | {'Params (M)':<10} | {'Size (MB)':<10} | {'CPU FPS':<10} | {'GPU FPS':<10}")
    print("-" * 65)
    
    results = []
    
    has_gpu = torch.cuda.is_available()
    
    for act in activations:
        model = YOLOLite(num_classes=cfg.num_classes, activation_name=act)
        
        params = count_parameters(model) / 1e6
        size_mb = get_model_size_mb(model)
        
        cpu_fps = benchmark_fps(model, torch.device('cpu'), cfg.input_size)
        
        gpu_fps = 0.0
        if has_gpu:
            gpu_fps = benchmark_fps(model, torch.device('cuda'), cfg.input_size)
            
        print(f"{act:<15} | {params:<10.2f} | {size_mb:<10.2f} | {cpu_fps:<10.1f} | {gpu_fps:<10.1f}")
        
        results.append([act, params, size_mb, cpu_fps, gpu_fps])
        
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Activation", "Params (M)", "Size (MB)", "CPU FPS", "GPU FPS"])
        writer.writerows(results)
        
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    run_benchmark()
