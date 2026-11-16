import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Elegant Dark Mode Palette matching TriageAI UI
DARK_BG = "#090d16"
CARD_BG = "#0e1626"
TEXT_COLOR = "#f1f5f9"
SUB_TEXT = "#94a3b8"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "savefig.facecolor": DARK_BG,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": "#233354",
    "xtick.color": SUB_TEXT,
    "ytick.color": SUB_TEXT,
    "grid.color": "#1e293b",
    "grid.alpha": 0.6,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

MODELS = [
    ("TF-IDF + Logistic Regression", "tfidf_logistic_regression"),
    ("TF-IDF + XGBoost", "tfidf_xgboost"),
    ("Sentence-BERT + XGBoost", "sbert_xgboost"),
    ("DistilBERT", "distilbert"),
    ("BERT", "bert"),
    ("Hybrid DistilBERT + SBERT", "hybrid"),
]

MODEL_COLORS = ["#a855f7", "#f59e0b", "#6366f1", "#ec4899", "#10b981", "#06b6d4"]


def plot_metric_comparison(df):
    """Plot grouped comparison of primary performance metrics."""
    metrics = ["Accuracy", "Weighted F1", "Macro F1", "Top-3 Accuracy"]
    available_metrics = [m for m in metrics if m in df.columns]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(df["Model"]))
    width = 0.18
    multiplier = 0

    palette = ["#6366f1", "#10b981", "#f59e0b", "#06b6d4"]
    for i, metric in enumerate(available_metrics):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            df[metric] * 100,
            width,
            label=metric,
            color=palette[i % len(palette)],
            alpha=0.9,
            edgecolor="#1e293b",
            linewidth=0.8,
            zorder=3
        )
        ax.bar_label(rects, padding=3, fmt="%.1f%%", fontsize=8, color="#f8fafc", rotation=0)
        multiplier += 1

    ax.set_ylabel("Score (%)", fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Performance Metrics Across All 6 Models (Held-out Test Split: 10,692 tickets)", fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
    ax.set_xticklabels(df["Model"], rotation=12, ha="right", fontweight="semibold")
    ax.legend(loc="upper left", facecolor=CARD_BG, edgecolor="#233354", labelcolor=TEXT_COLOR)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plot_path = PLOTS_DIR / "model_metrics_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_efficiency_tradeoff(df):
    """Plot Training Time vs Accuracy (Tradeoff Analysis)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, row in df.iterrows():
        train_time = row["Training Time (seconds)"]
        acc = row["Accuracy"] * 100
        if pd.isna(train_time) or train_time is None:
            continue
        ax.scatter(
            train_time,
            acc,
            s=220,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            edgecolors="#fff",
            linewidth=1.5,
            zorder=5
        )
        ax.annotate(
            row["Model"],
            (train_time, acc),
            xytext=(10, 6),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9.5,
            color=TEXT_COLOR,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (seconds, log scale)", fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Accuracy vs. Training Time Efficiency Frontier", fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="#1e293b", zorder=1)

    plot_path = PLOTS_DIR / "training_vs_accuracy_tradeoff.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_inference_latency(df):
    """Plot prediction time per 10k tickets."""
    fig, ax = plt.subplots(figsize=(10, 5))

    pred_times = df["Prediction Time (seconds)"]
    bars = ax.barh(df["Model"], pred_times, color="#38bdf8", edgecolor="#0284c7", linewidth=0.8, height=0.52, zorder=3)
    ax.bar_label(bars, padding=6, fmt="%.3fs", fontweight="semibold", color="#f8fafc", fontsize=9.5)

    ax.set_xlabel("Inference Time on Test Set (10,692 tickets in seconds)", fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Inference Latency Benchmark (Lower is Faster)", fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#1e293b", zorder=0)

    plot_path = PLOTS_DIR / "inference_latency_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_dataset_queue_distribution():
    """Plot top queue class distributions from cleaned tickets."""
    csv_path = ROOT / "data" / "cleaned_tickets.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, low_memory=False)
    counts = df["queue"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(x=counts.values, y=counts.index, palette="mako", ax=ax, edgecolor="#1e293b", linewidth=0.6, zorder=3)
    ax.set_xlabel("Number of Tickets", fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Support Queue Category", fontweight="bold", color=TEXT_COLOR)
    ax.set_title(f"Top 15 Support Ticket Queues (out of {df['queue'].nunique()} total classes)", fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#1e293b", zorder=0)

    plot_path = PLOTS_DIR / "queue_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_confusion_matrices():
    """Plot heatmaps of confusion matrices for available models."""
    available_cms = []
    for display_name, stem in MODELS:
        cm_path = RESULTS_DIR / f"{stem}_confusion_matrix.csv"
        if cm_path.exists():
            cm = np.loadtxt(cm_path, delimiter=",")
            available_cms.append((display_name, cm))

    if not available_cms:
        return

    n_models = len(available_cms)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), squeeze=False, facecolor=DARK_BG)

    for idx, (name, cm) in enumerate(available_cms):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        ax.set_facecolor(CARD_BG)

        # Normalize by true class
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_cm = cm / cm.sum(axis=1, keepdims=True)
            norm_cm = np.nan_to_num(norm_cm)

        im = ax.imshow(norm_cm, cmap="mako", interpolation="nearest", aspect="auto")
        ax.set_title(name, fontsize=10.5, fontweight="bold", color=TEXT_COLOR)
        ax.set_xlabel("Predicted Class (0-51)", fontsize=8.5, color=SUB_TEXT)
        ax.set_ylabel("True Class (0-51)", fontsize=8.5, color=SUB_TEXT)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=SUB_TEXT)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=SUB_TEXT, size=8)

    # Hide extra subplots
    for idx in range(n_models, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")

    plt.suptitle("Normalized Confusion Matrices Across Models (52 Support Queues)", fontsize=14, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plot_path = PLOTS_DIR / "confusion_matrices_grid.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    rows = []
    for display_name, stem in MODELS:
        res_file = RESULTS_DIR / f"{stem}_results.json"
        if not res_file.exists():
            continue
        with res_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "Model": display_name,
            "Accuracy": data.get("accuracy"),
            "Weighted Precision": data.get("precision"),
            "Weighted Recall": data.get("recall"),
            "Weighted F1": data.get("f1") or data.get("weighted_f1"),
            "Macro F1": data.get("macro_f1"),
            "Top-3 Accuracy": data.get("top_3_accuracy"),
            "ROC-AUC (OVR)": data.get("roc_auc_ovr_weighted"),
            "Training Time (seconds)": data.get("training_time_seconds"),
            "Prediction Time (seconds)": data.get("prediction_time_seconds"),
            "Embedding Time (seconds)": data.get("embedding_time_seconds"),
            "Retrieval Hit@3": data.get("retrieval_hit_at_3"),
        })

    if not rows:
        print("No results found yet.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "comprehensive_model_comparison.csv", index=False)
    print("Generating sleek dark-theme visual plots...")
    plot_metric_comparison(df)
    plot_efficiency_tradeoff(df)
    plot_inference_latency(df)
    plot_dataset_queue_distribution()
    plot_confusion_matrices()
    print("All visualizations successfully created!")


if __name__ == "__main__":
    main()
