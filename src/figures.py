"""
figures.py — 학습 곡선 시각화
train()이 반환하는 history dict를 받아 loss/accuracy 곡선 저장
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt


BG         = "#12121f"
TEXT_COLOR = "white"
C_TRAIN    = "#4fc3f7"   # 파랑
C_VAL      = "#f48fb1"   # 분홍
C_GATE     = "#a5d6a7"   # 연초록


def plot_training_curves(history: dict, save_path: str, title: str = "Training"):
    """
    train()이 반환한 history를 받아 3-panel 학습 곡선 저장.

    Panels:
      1. Train / Val Loss
      2. Train / Val Accuracy
      3. Gate Loss (gate 없으면 빈 칸)

    Args:
        history   : train() 반환 dict
        save_path : 저장할 .png 경로
        title     : 그래프 제목 prefix
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = list(range(len(history["train_loss"])))
    stopped = history.get("stopped_epoch", epochs[-1] if epochs else 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ── Panel 1: Loss ─────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=C_TRAIN, linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=C_VAL,   linewidth=2, label="Val Loss")
    ax.axvline(stopped, color="#888", linestyle="--", linewidth=1, label=f"Stop @{stopped}")
    ax.set_title("Loss", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Loss",  color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=9)

    # ── Panel 2: Accuracy ─────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=C_TRAIN, linewidth=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"],   color=C_VAL,   linewidth=2, label="Val Acc")
    ax.axvline(stopped, color="#888", linestyle="--", linewidth=1)
    best = history.get("best_val_acc", max(history["val_acc"]) if history["val_acc"] else 0)
    ax.set_title(f"Accuracy  (best val={best:.4f})", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Acc",   color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=9)

    # ── Panel 3: Gate Loss ────────────────────────────────
    ax = axes[2]
    gate_vals = history.get("gate_loss", [])
    if gate_vals and max(gate_vals) > 1e-8:
        ax.plot(epochs, gate_vals, color=C_GATE, linewidth=2, label="Gate Loss")
        ax.axvline(stopped, color="#888", linestyle="--", linewidth=1)
        ax.set_title("Gate Loss", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Gate\n(Baseline)", ha="center", va="center",
                color="#666", fontsize=14, transform=ax.transAxes)
        ax.set_title("Gate Loss (N/A)", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Loss",  color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=9)

    fig.suptitle(title, color=TEXT_COLOR, fontsize=14, fontweight="bold")
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[figures] 저장: {save_path}")


def plot_comparison_curves(baseline_hist: dict, gated_hist: dict, save_path: str):
    """
    Baseline vs Gated 학습 곡선 비교 (val loss / val acc 2-panel).

    Args:
        baseline_hist : baseline train() history
        gated_hist    : gated train() history
        save_path     : 저장 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    e_base  = list(range(len(baseline_hist["val_loss"])))
    e_gated = list(range(len(gated_hist["val_loss"])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Val Loss
    ax = axes[0]
    ax.plot(e_base,  baseline_hist["val_loss"], color=C_TRAIN, linewidth=2, label="Baseline Val Loss")
    ax.plot(e_gated, gated_hist["val_loss"],    color=C_VAL,   linewidth=2, label="Gated Val Loss")
    ax.set_title("Validation Loss", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Loss",  color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=9)

    # Val Acc
    ax = axes[1]
    ax.plot(e_base,  baseline_hist["val_acc"], color=C_TRAIN, linewidth=2, label="Baseline Val Acc")
    ax.plot(e_gated, gated_hist["val_acc"],    color=C_VAL,   linewidth=2, label="Gated Val Acc")
    ax.set_title("Validation Accuracy", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Acc",   color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=9)

    fig.suptitle("Baseline vs Gated — Training Comparison", color=TEXT_COLOR, fontsize=14, fontweight="bold")
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[figures] 저장: {save_path}")


def save_history_json(history: dict, path: str):
    """history dict를 JSON으로 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # numpy/tensor → python float 변환
    clean = {}
    for k, v in history.items():
        if isinstance(v, list):
            clean[k] = [float(x) for x in v]
        else:
            clean[k] = float(v) if isinstance(v, (int, float, np.floating)) else v

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[figures] 저장: {path}")
