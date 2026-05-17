"""
visualize.py — Visualization Utilities
======================================

Generates plots for the ablation study:
- Loss curves
- FPS comparison
- Activation function shapes
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from yololite.activations import get_activation

def plot_activation_functions(save_dir="logs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    x = torch.linspace(-5, 5, 500)
    
    activations = ["relu", "leaky", "silu", "mish", "hardswish"]
    
    plt.figure(figsize=(10, 6))
    
    for act_name in activations:
        act_fn = get_activation(act_name)
        y = act_fn(x.clone()).detach().numpy()
        plt.plot(x.numpy(), y, label=act_name.capitalize(), linewidth=2)
        
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.title("Activation Functions Used in YOLO-LITE Ablation Study", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    
    save_path = os.path.join(save_dir, "activation_functions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_training_curves(log_dir="logs", save_dir="logs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    activations = ["relu", "leaky", "silu", "mish", "hardswish"]
    
    plt.figure(figsize=(12, 6))
    
    for act in activations:
        csv_path = os.path.join(log_dir, f"{act}_run.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_loss'], label=act.capitalize(), linewidth=2)
            
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.title("Validation Loss Comparison", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    
    save_path = os.path.join(save_dir, "val_loss_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_benchmark_results(csv_path="logs/benchmark_results.csv", save_dir="logs/plots"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Plot CPU FPS
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Activation'], df['CPU FPS'], color='skyblue', edgecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 1), ha='center', va='bottom', fontsize=11)
        
    plt.grid(axis='y', alpha=0.3)
    plt.title("CPU Inference Speed Comparison (FPS)", fontsize=14)
    plt.xlabel("Activation Function", fontsize=12)
    plt.ylabel("Frames Per Second", fontsize=12)
    
    save_path = os.path.join(save_dir, "cpu_fps_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_activation_functions()
    plot_training_curves()
    plot_benchmark_results()
