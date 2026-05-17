import subprocess
import sys
import os

def run_all_experiments():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    # ensure logs dir exists
    os.makedirs("logs", exist_ok=True)

    scripts = [
        ("experiments/run_baseline_leaky.py", "logs/run_baseline_leaky.log"),
        ("experiments/run_baseline_relu.py", "logs/run_baseline_relu.log"),
        ("experiments/run_hardswish.py", "logs/run_hardswish.log"),
        ("experiments/run_mish.py", "logs/run_mish.log"),
        ("experiments/run_silu.py", "logs/run_silu.log"),
    ]

    processes = []
    
    print("Starting all experiments in parallel...")
    for script, log_file in scripts:
        print(f"Starting {script}, logging output to {log_file}")
        f = open(log_file, "w")
        p = subprocess.Popen([sys.executable, script], stdout=f, stderr=subprocess.STDOUT)
        processes.append((p, f, script))

    print("All experiments have been launched. Waiting for them to finish...")
    
    for p, f, script in processes:
        p.wait()
        f.close()
        if p.returncode == 0:
            print(f"✅ {script} finished successfully.")
        else:
            print(f"❌ {script} failed with return code {p.returncode}. Check its log file.")
            
    print("All experiments completed!")

if __name__ == "__main__":
    run_all_experiments()
