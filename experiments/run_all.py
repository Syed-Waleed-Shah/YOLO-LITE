import subprocess
import sys
import os
import threading

def stream_output(process, log_file, script_name):
    # Extract just the basename without extension for cleaner terminal output
    short_name = os.path.basename(script_name).replace('.py', '').replace('run_', '')
    with open(log_file, "w", encoding="utf-8") as f:
        for line in iter(process.stdout.readline, b''):
            line_str = line.decode('utf-8', errors='replace')
            f.write(line_str)
            f.flush()
            
            # Print epoch progress to terminal
            if line_str.startswith("Epoch ") and "Train Loss" in line_str:
                print(f"[{short_name}] {line_str.strip()}")
            elif "--- Starting Training Run" in line_str or "Device:" in line_str:
                print(f"[{short_name}] {line_str.strip()}")

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
    threads = []
    
    print("Starting all experiments in parallel...")
    for script, log_file in scripts:
        print(f"Starting {script}, logging output to {log_file}")
        p = subprocess.Popen(
            [sys.executable, script], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT
        )
        t = threading.Thread(target=stream_output, args=(p, log_file, script))
        t.start()
        processes.append((p, script))
        threads.append(t)

    print("All experiments have been launched. Waiting for them to finish...")
    
    for (p, script), t in zip(processes, threads):
        p.wait()
        t.join()
        if p.returncode == 0:
            print(f"✅ {script} finished successfully.")
        else:
            print(f"❌ {script} failed with return code {p.returncode}. Check its log file.")
            
    print("All experiments completed!")

if __name__ == "__main__":
    run_all_experiments()
