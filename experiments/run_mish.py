import subprocess
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
sys.path.append(project_root)

print("Starting Mish training run...")
subprocess.run([
    sys.executable, "train.py",
    "--config", "experiments/configs/mish.yaml"
])
