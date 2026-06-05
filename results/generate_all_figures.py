import os
import subprocess
import sys

# Paths
RESULTS_DIR = r"d:\EEG\results"
WORKSPACE_DIR = r"d:\EEG"

def run_script(script_path, cwd=WORKSPACE_DIR):
    print(f"\n>>> Running: {script_path} (cwd: {cwd})")
    python_bin = sys.executable
    res = subprocess.run([python_bin, script_path], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing {script_path}:")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        return False
    else:
        print(f"Success!")
        print("STDOUT:", res.stdout)
        return True

def main():
    print("=========================================================")
    print("      BCI PAPER FIGURES GENERATION ORCHESTRATOR           ")
    print("=========================================================")
    
    # 1. Generate Figures 1, 2, 3, 5 (schematics)
    schematic_script = os.path.join(RESULTS_DIR, "generate_schematic_figures.py")
    if not run_script(schematic_script):
        print("[ERROR] Failed to generate schematic figures.")
        sys.exit(1)
        
    # 2. Generate Figures 7 and 8 (comparative evaluation & trajectory) by running benchmarking.py
    benchmarking_script = os.path.join(WORKSPACE_DIR, "src", "benchmarking.py")
    if not run_script(benchmarking_script):
        print("[ERROR] Failed to run benchmarking script.")
        sys.exit(1)
        
    # 3. Verify files
    expected_files = [
        "conceptual_overview.png",
        "research_workflow.png",
        "system_architecture.png",
        "experimental_setup.jpg",
        "stimulus_timeline.png",
        "1D_cnn_accuracy_curve.png",
        "LSTM_confusion_matrix_eeg_seq_classification.png",
        "comparative_confusion_matrices.png",
        "trajectory_comparison.png"
    ]
    
    print("\n================ Verification Summary ================")
    all_present = True
    for f in expected_files:
        path = os.path.join(RESULTS_DIR, f)
        exists = os.path.exists(path)
        status = "[OK] Present" if exists else "[ERROR] Missing"
        if not exists:
            all_present = False
        print(f"{f:<50} : {status}")
        
    if all_present:
        print("\nAll figures (Figures 1 to 8) generated and verified successfully!")
        sys.exit(0)
    else:
        print("\nSome figures are missing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
