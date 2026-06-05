import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure results directory exists
RESULTS_DIR = r"d:\EEG\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set global matplotlib styling for academic look
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#2c3e50'
plt.rcParams['axes.labelcolor'] = '#2c3e50'
plt.rcParams['xtick.color'] = '#2c3e50'
plt.rcParams['ytick.color'] = '#2c3e50'

def draw_figure1():
    """Draws Figure 1: Conceptual Overview"""
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    
    # 1. Colors
    col_problem = '#e74c3c'  # Soft red
    col_solution = '#2ecc71' # Soft green
    col_neutral = '#3498db'  # Soft blue
    col_text = '#2c3e50'
    col_bg_prob = '#fdf2f2'
    col_bg_sol = '#f2fcf5'
    col_bg_neu = '#f2f7fc'
    
    # Title
    ax.text(50, 47, "Figure 1: Conceptual Pipeline & Target Resolution", 
            fontsize=15, fontweight='bold', ha='center', color=col_text)
    
    # Draw Problems (Left side)
    problems = [
        {"name": "Noise", "desc": "Low SNR, ocular/\nmuscle artifacts", "y": 32},
        {"name": "Data Scarcity", "desc": "Expensive, fatiguing\nsubject labeling", "y": 20},
        {"name": "Deployment Drift", "desc": "Electrode drift &\nsession-to-session shift", "y": 8}
    ]
    
    for p in problems:
        # Box
        rect = patches.FancyBboxPatch((5, p["y"]-4), 22, 8, boxstyle="round,pad=1.5",
                                     edgecolor=col_problem, facecolor=col_bg_prob, lw=2)
        ax.add_patch(rect)
        ax.text(16, p["y"]+1.5, p["name"], fontsize=11, fontweight='bold', ha='center', color=col_problem)
        ax.text(16, p["y"]-2, p["desc"], fontsize=9.5, ha='center', color=col_text)
        
    # Draw Pipeline (Right side horizontal flow)
    stages = [
        {"name": "Raw EEG\nInput", "x": 37, "y": 20, "w": 10, "h": 8, "bg": col_bg_neu, "ec": col_neutral},
        {"name": "Bipolar &\nChebyshev", "x": 51, "y": 20, "w": 11, "h": 8, "bg": col_bg_sol, "ec": col_solution},
        {"name": "Spectral\nWGAN-GP", "x": 66, "y": 20, "w": 11, "h": 8, "bg": col_bg_sol, "ec": col_solution},
        {"name": "Decoder\n(CNN/LSTM)", "x": 81, "y": 20, "w": 11, "h": 8, "bg": col_bg_neu, "ec": col_neutral},
        {"name": "Adaptive Calib. &\nDecision Fusion", "x": 55, "y": 5, "w": 25, "h": 8, "bg": col_bg_sol, "ec": col_solution},
    ]
    
    for s in stages:
        rect = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0.5",
                                     edgecolor=s["ec"], facecolor=s["bg"], lw=2)
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2.0, s["y"] + s["h"]/2.0, s["name"], fontsize=9.5, 
                fontweight='bold', ha='center', va='center', color=col_text)
        
    # Connections in Pipeline
    ax.annotate("", xy=(51, 24), xytext=(47, 24), arrowprops=dict(arrowstyle="->", color=col_text, lw=1.5))
    ax.annotate("", xy=(66, 24), xytext=(62, 24), arrowprops=dict(arrowstyle="->", color=col_text, lw=1.5))
    ax.annotate("", xy=(81, 24), xytext=(77, 24), arrowprops=dict(arrowstyle="->", color=col_text, lw=1.5))
    
    # Decoder output split
    ax.annotate("", xy=(93, 24), xytext=(96, 24), arrowprops=dict(arrowstyle="<-", color=col_text, lw=1.5))
    ax.text(97, 24, "Control\nCommand", fontsize=9, fontweight='bold', color=col_text, va='center')
    
    # Back-connect from Decoder to calibration
    ax.annotate("", xy=(67.5, 13), xytext=(86.5, 20),
                arrowprops=dict(arrowstyle="->", color=col_text, lw=1.5, connectionstyle="angle,angleA=90,angleB=180,rad=5"))
    
    # Connect calibration to Command
    ax.annotate("", xy=(95, 20), xytext=(80, 9),
                arrowprops=dict(arrowstyle="->", color=col_text, lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    
    # 3. Draw problem-resolution arrows
    # Noise -> Bipolar & Chebyshev
    ax.annotate("Resolves Noise", xy=(56.5, 29), xytext=(28, 32),
                arrowprops=dict(arrowstyle="-|>", color=col_solution, lw=2, ls='--'),
                fontsize=9, color=col_solution, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=col_solution, lw=1))
    
    # Data Scarcity -> WGAN-GP
    ax.annotate("Resolves Scarcity", xy=(71.5, 29), xytext=(28, 20),
                arrowprops=dict(arrowstyle="-|>", color=col_solution, lw=2, ls='--'),
                fontsize=9, color=col_solution, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=col_solution, lw=1))
    
    # Deployment Drift -> Adaptive Calib
    ax.annotate("Resolves Session Drift", xy=(55, 9), xytext=(28, 8),
                arrowprops=dict(arrowstyle="-|>", color=col_solution, lw=2, ls='--'),
                fontsize=9, color=col_solution, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=col_solution, lw=1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "conceptual_overview.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated conceptual_overview.png")

def draw_figure2():
    """Draws Figure 2: End-to-end Research Workflow"""
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    col_phase = '#34495e'
    col_bg = '#f8f9f9'
    col_arrow = '#7f8c8d'
    
    ax.text(50, 96, "Figure 2: End-to-End BCI Research Workflow", fontsize=15, fontweight='bold', ha='center')
    
    # Define phases
    phases = [
        {
            "num": "Phase 1",
            "title": "Dataset Conditioning",
            "bullets": [
                "Bipolar Subtractions: P4-O2, P3-O1, F4-C4",
                "6th-order Chebyshev band-pass (8-30 Hz)",
                "File-level train-test split (80/20)"
            ],
            "x": 5, "y": 66, "w": 40, "h": 22
        },
        {
            "num": "Phase 2",
            "title": "Generative Data Augmentation",
            "bullets": [
                "1D-FFT L1 spectral loss matching",
                "WGAN-GP adversarial time-series training",
                "Overlap-Add window stitching (stride=32)"
            ],
            "x": 55, "y": 66, "w": 40, "h": 22
        },
        {
            "num": "Phase 3",
            "title": "Classifier Training",
            "bullets": [
                "Pure 1D-CNN (3 conv layers, kernel 7-5-3)",
                "Hybrid ConvLSTM (CNN front, 128 hidden LSTM)",
                "AdamW optimizer, 20 epochs, scheduling"
            ],
            "x": 55, "y": 36, "w": 40, "h": 22
        },
        {
            "num": "Phase 4",
            "title": "Comparative Benchmarking",
            "bullets": [
                "Unseen cross-session test files (LE, RY, For)",
                "Branch A: Static global training scaling",
                "Branch B: Dynamic local active session scaling"
            ],
            "x": 5, "y": 36, "w": 40, "h": 22
        },
        {
            "num": "Phase 5",
            "title": "Safety Gating & Decision Fusion",
            "bullets": [
                "Confidence Stability Margin: ΔP = P(1) - P(2)",
                "Flags shifting states when margin < 15%",
                "Majority mode voting over sliding window"
            ],
            "x": 5, "y": 6, "w": 40, "h": 22
        },
        {
            "num": "Phase 6",
            "title": "Validation & Paper Draft",
            "bullets": [
                "Output trajectory and class collapse analyses",
                "Confusion matrices & Wilcoxon tests",
                "Final academic paper tables and reports"
            ],
            "x": 55, "y": 6, "w": 40, "h": 22
        }
    ]
    
    for p in phases:
        # Box
        rect = patches.FancyBboxPatch((p["x"], p["y"]), p["w"], p["h"], boxstyle="round,pad=1.0",
                                     edgecolor=col_phase, facecolor=col_bg, lw=2)
        ax.add_patch(rect)
        
        # Header
        ax.text(p["x"]+2, p["y"]+p["h"]-2.5, f"{p['num']}: {p['title']}", 
                fontsize=10.5, fontweight='bold', color='#2c3e50')
        
        # Bullets
        by = p["y"] + p["h"] - 6
        for b in p["bullets"]:
            ax.text(p["x"]+2, by, f"• {b}", fontsize=8.5, color='#34495e')
            by -= 4.2
            
    # Connective Arrows
    # P1 -> P2
    ax.annotate("", xy=(55, 77), xytext=(45, 77), arrowprops=dict(arrowstyle="-|>", color=col_arrow, lw=2))
    # P2 -> P3
    ax.annotate("", xy=(75, 58), xytext=(75, 66), arrowprops=dict(arrowstyle="-|>", color=col_arrow, lw=2))
    # P3 -> P4
    ax.annotate("", xy=(45, 47), xytext=(55, 47), arrowprops=dict(arrowstyle="-|>", color=col_arrow, lw=2))
    # P4 -> P5
    ax.annotate("", xy=(25, 28), xytext=(25, 36), arrowprops=dict(arrowstyle="-|>", color=col_arrow, lw=2))
    # P5 -> P6
    ax.annotate("", xy=(55, 17), xytext=(45, 17), arrowprops=dict(arrowstyle="-|>", color=col_arrow, lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "research_workflow.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated research_workflow.png")

def draw_figure3():
    """Draws Figure 3: System Architecture"""
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    col_title = '#2c3e50'
    col_wgan = '#9b59b6'  # Purple
    col_dec = '#2980b9'   # Blue
    col_cal = '#27ae60'   # Green
    col_bg = '#fcf8ff'
    
    ax.text(50, 97, "Figure 3: System Architecture & Sub-module Flow", fontsize=15, fontweight='bold', ha='center', color=col_title)
    
    # ------------------- BLOCK A: WGAN-GP (Top) -------------------
    # Border
    rect_a = patches.Rectangle((2, 50), 96, 43, fill=True, facecolor='#faf8fc', edgecolor=col_wgan, lw=1.5, ls='--')
    ax.add_patch(rect_a)
    ax.text(4, 90.5, "A. Spectral-Constrained WGAN-GP Augmentation", fontsize=11, fontweight='bold', color=col_wgan)
    
    # Generator
    ax.add_patch(patches.FancyBboxPatch((4, 62), 24, 23, boxstyle="round,pad=0.5", edgecolor=col_wgan, facecolor='#fff', lw=1.5))
    ax.text(16, 82.5, "GENERATOR (G)", fontsize=9.5, fontweight='bold', ha='center', color=col_wgan)
    ax.text(16, 78, "Latent z ~ N(0, I) [100]\n"
                    "  ↓\n"
                    "FC (256x16)\n"
                    "  ↓\n"
                    "3x ConvTranspose1d\n"
                    "(256→128→64→32, BN+ReLU)\n"
                    "  ↓\n"
                    "Conv1d (channels=3, kernel=1)\n"
                    "  ↓\n"
                    "Generated Window [3x128]", fontsize=8, ha='center')
    
    # Critic
    ax.add_patch(patches.FancyBboxPatch((42, 62), 24, 23, boxstyle="round,pad=0.5", edgecolor=col_wgan, facecolor='#fff', lw=1.5))
    ax.text(54, 82.5, "CRITIC (C)", fontsize=9.5, fontweight='bold', ha='center', color=col_wgan)
    ax.text(54, 78, "Input Window [3x128]\n"
                    "(Real OR Generated)\n"
                    "  ↓\n"
                    "3x Conv1d Blocks\n"
                    "(kernel=5, LeakyReLU)\n"
                    "  ↓\n"
                    "Flatten & FC Linear\n"
                    "  ↓\n"
                    "Wasserstein Score (Scalar)\n"
                    "  ↓\n"
                    "1-Lipschitz (Gradient Penalty)", fontsize=8, ha='center')
    
    # Spectral Loss Block
    ax.add_patch(patches.FancyBboxPatch((72, 65), 22, 17, boxstyle="round,pad=0.5", edgecolor=col_wgan, facecolor='#fff', lw=1.5))
    ax.text(83, 79.5, "Spectral Constraint", fontsize=9.5, fontweight='bold', ha='center', color=col_wgan)
    ax.text(83, 70, "1D-FFT Magnitude Loss\n"
                    "L_spec = L1(|RFFT_real|\n"
                    " - |RFFT_fake|)\n\n"
                    "Stitched via Overlap-Add\n"
                    "to save continuous files.", fontsize=8, ha='center')
    
    # Arrows inside WGAN
    ax.annotate("Synthetic data", xy=(42, 73.5), xytext=(28, 73.5), arrowprops=dict(arrowstyle="-|>", color=col_wgan, lw=1.5))
    ax.annotate("Real data", xy=(42, 78.5), xytext=(35, 78.5), arrowprops=dict(arrowstyle="-|>", color=col_wgan, lw=1.5))
    ax.annotate("FFT regularizer", xy=(72, 73.5), xytext=(66, 73.5), arrowprops=dict(arrowstyle="-|>", color=col_wgan, lw=1.5))
    
    # ------------------- BLOCK B: Decoders (Bottom Left) -------------------
    rect_b = patches.Rectangle((2, 2), 46, 45, fill=True, facecolor='#f5fafe', edgecolor=col_dec, lw=1.5, ls='--')
    ax.add_patch(rect_b)
    ax.text(4, 44, "B. Spatial-Temporal Decoders", fontsize=11, fontweight='bold', color=col_dec)
    
    # 1D-CNN
    ax.add_patch(patches.FancyBboxPatch((4, 25), 42, 16, boxstyle="round,pad=0.5", edgecolor=col_dec, facecolor='#fff', lw=1.5))
    ax.text(25, 38.5, "Pure 1D-CNN", fontsize=9.5, fontweight='bold', ha='center', color=col_dec)
    ax.text(25, 27.5, "Input: [3 x 256] EEG window\n"
                    "3x Conv Blocks (kernel=7/5/3, width=64/128/256, BN+ReLU+MaxPool+Drop)\n"
                    "AdaptiveAvgPool1d(1) → Flatten\n"
                    "FC Classifier Head (256 → 64 → 4 Classes)", fontsize=8, ha='center')
    
    # ConvLSTM
    ax.add_patch(patches.FancyBboxPatch((4, 5), 42, 16, boxstyle="round,pad=0.5", edgecolor=col_dec, facecolor='#fff', lw=1.5))
    ax.text(25, 18.5, "Hybrid ConvLSTM", fontsize=9.5, fontweight='bold', ha='center', color=col_dec)
    ax.text(25, 7.5, "Input: [3 x 256] EEG window\n"
                    "2x 1D-CNN Front-End layers (kernel=5/3, 64 ch, BN+MaxPool+Drop)\n"
                    "2-layer LSTM (128 hidden state) -> Last time step\n"
                    "FC Classifier Head (128 → 64 → 4 Classes)", fontsize=8, ha='center')
    
    # ------------------- BLOCK C: Adaptive Calibration & Fusion (Bottom Right) -------------------
    rect_c = patches.Rectangle((52, 2), 46, 45, fill=True, facecolor='#f4fbf7', edgecolor=col_cal, lw=1.5, ls='--')
    ax.add_patch(rect_c)
    ax.text(54, 44, "C. Adaptive Calibration & Decision Fusion", fontsize=11, fontweight='bold', color=col_cal)
    
    ax.add_patch(patches.FancyBboxPatch((54, 5), 42, 36, boxstyle="round,pad=0.5", edgecolor=col_cal, facecolor='#fff', lw=1.5))
    ax.text(75, 38.5, "Calibration & Stabilization Engine", fontsize=9.5, fontweight='bold', ha='center', color=col_cal)
    ax.text(75, 10, "1. Linear Detrending\n"
                    "   Removes low-frequency baseline drift from signals\n"
                    "   X_det = detrend(X)\n\n"
                    "2. Dynamic Active Session Normalization\n"
                    "   Uses session-specific mean and scale parameters\n"
                    "   X_cal = (X_det - mean_ses) / std_ses\n\n"
                    "3. Confidence Stability Margin Gating\n"
                    "   Sorted probabilities: P(1) >= P(2) >= ...\n"
                    "   Delta P = P(1) - P(2)\n"
                    "   If Delta P < 0.15 → Block prediction (SHIFTING)\n"
                    "   Else → Accept prediction (STABLE)\n\n"
                    "4. Majority Mode Voting\n"
                    "   Aggregates STABLE predictions → Outbound Command", fontsize=8.2, ha='center')
    
    # Inter-block flow
    ax.annotate("Trained weights", xy=(4, 33), xytext=(2, 60), 
                arrowprops=dict(arrowstyle="->", color=col_dec, lw=1.5, connectionstyle="arc3,rad=-0.2"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "system_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated system_architecture.png")

def draw_figure5():
    """Draws Figure 5: Stimulus presentation timeline"""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    
    # Draw timeline line
    ax.axhline(y=1, color='#7f8c8d', lw=2, zorder=1)
    
    # Draw tick marks and blocks
    # Blocks:
    # 1. Fixation: 0 - 2s
    # 2. Cue: 2 - 3s
    # 3. Intent: 3 - 7s
    # 4. Rest: 7 - 9s
    
    blocks = [
        {"name": "Fixation Cross", "start": 0, "end": 2, "color": "#bdc3c7", "desc": "Baseline\n[+] cross screen", "icon": "+"},
        {"name": "Directional Cue", "start": 2, "end": 3, "color": "#e67e22", "desc": "ARROW / LETTER / WORD\n(Target intent direction)", "icon": "→"},
        {"name": "Motor Intent Window", "start": 3, "end": 7, "color": "#2ecc71", "desc": "Sustained motor imagery\nContinuous EEG recorded", "icon": "EEG"},
        {"name": "Rest Period", "start": 7, "end": 9, "color": "#ecf0f1", "desc": "Inter-trial rest\nRelaxation interval", "icon": "zZz"}
    ]
    
    for b in blocks:
        # Draw block area
        rect = patches.Rectangle((b["start"], 1.05), b["end"] - b["start"], 0.6,
                                 facecolor=b["color"], edgecolor='#34495e', alpha=0.9, zorder=2)
        ax.add_patch(rect)
        
        # Center text label inside block
        ax.text(b["start"] + (b["end"]-b["start"])/2.0, 1.45, b["name"], 
                fontsize=10.5, fontweight='bold', ha='center', color='#2c3e50')
        ax.text(b["start"] + (b["end"]-b["start"])/2.0, 1.25, f"Icon: {b['icon']}" if b["name"] == "Rest Period" else b["icon"], 
                fontsize=13, fontweight='bold', ha='center', color='#2c3e50', alpha=0.5)
        
        # Description text below timeline
        ax.text(b["start"] + (b["end"]-b["start"])/2.0, 0.7, b["desc"], 
                fontsize=8.5, ha='center', va='top', color='#34495e')
        
        # Duration label
        duration = b["end"] - b["start"]
        ax.text(b["start"] + (b["end"]-b["start"])/2.0, 1.7, f"{duration:.1f} s", 
                fontsize=9.5, fontweight='semibold', ha='center', color='#2c3e50')
        
        # Draw timeline boundary ticks
        ax.axvline(x=b["start"], ymin=0.45, ymax=0.55, color='#34495e', lw=1.5, zorder=3)
        ax.text(b["start"], 0.9, f"{b['start']:.1f}s", fontsize=8.5, ha='center', color='#7f8c8d')
        
    # Final tick mark
    ax.axvline(x=9, ymin=0.45, ymax=0.55, color='#34495e', lw=1.5, zorder=3)
    ax.text(9, 0.9, "9.0s", fontsize=8.5, ha='center', color='#7f8c8d')
    
    # Styling and limits
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0.3, 1.9)
    ax.axis('off')
    
    # Repeated annotation at the bottom
    ax.text(4.5, 0.4, "Interleaved Paradigm: Cue modalities (ARROW / LETTER / WORD) randomized across trials\n"
                      "Repeated for 20 trials per directional class.", 
            fontsize=9.5, fontstyle='italic', ha='center', color='#7f8c8d',
            bbox=dict(boxstyle="round,pad=0.5", fc='#fafafa', ec='#bdc3c7', lw=1))
    
    ax.set_title("Figure 5: Single-Trial Stimulus Presentation Paradigm Timeline", 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stimulus_timeline.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated stimulus_timeline.png")

if __name__ == "__main__":
    print("Generating schematic figures...")
    draw_figure1()
    draw_figure2()
    draw_figure3()
    draw_figure5()
    print("All schematic figures generated successfully!")
