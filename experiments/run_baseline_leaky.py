import subprocess
import sys
import os

# Ensure we're in the correct root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)

# Setup path so yololite can be imported
sys.path.append(project_root)

print("Starting Baseline (LeakyReLU) training run...")
subprocess.run([
    sys.executable, "train.py",
    "--config", "experiments/configs/baseline_leaky.yaml"
])
