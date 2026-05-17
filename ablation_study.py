"""
ablation_study.py — Generates formatted ablation study tables from CSV logs
"""

import pandas as pd
import os

def generate_ablation_table(log_dir="logs", output_file="logs/ablation_table.md"):
    activations = ["leaky", "relu", "silu", "mish", "hardswish"]
    
    benchmark_path = os.path.join(log_dir, "benchmark_results.csv")
    bench_data = None
    if os.path.exists(benchmark_path):
        bench_data = pd.read_csv(benchmark_path).set_index("Activation")
        
    results = []
    
    for act in activations:
        csv_path = os.path.join(log_dir, f"{act}_run.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            best_epoch = df['val_loss'].idxmin()
            best_row = df.loc[best_epoch]
            
            cpu_fps = "N/A"
            if bench_data is not None and act in bench_data.index:
                cpu_fps = f"{bench_data.loc[act, 'CPU FPS']:.1f}"
                
            results.append({
                "Variant": act.capitalize(),
                "Val Loss": f"{best_row['val_loss']:.4f}",
                "mAP@0.5": f"{best_row['mAP']:.4f}",
                "CPU FPS": cpu_fps
            })
            
    if not results:
        print("No CSV logs found to generate ablation table.")
        return
        
    res_df = pd.DataFrame(results)
    
    markdown_table = res_df.to_markdown(index=False)
    
    with open(output_file, 'w') as f:
        f.write("# YOLO-LITE Activation Ablation Study Results\n\n")
        f.write("Baseline model corresponds to 'Leaky' (LeakyReLU with alpha=0.1).\n\n")
        f.write(markdown_table)
        f.write("\n\n*Note: mAP@0.5 and validation loss reflect best-epoch performance.*")
        
    print(f"Ablation table saved to {output_file}")
    print("\n" + markdown_table)

if __name__ == "__main__":
    generate_ablation_table()
