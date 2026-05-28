import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EnzymeSubstrateDataset, collate_fn
from src.metrics import compute_regression_metrics
from src.models import HeteroscedasticEnzymeModel

# =============================================================================
# Global configuration
# =============================================================================
TASKS = ["kcat", "km", "ki"]
MODALITIES = ["seq_only", "bimodal", "trimodal"]
METRIC_NAMES = ["R2", "Pearson", "RMSE", "MAE"]
PICTURES_DIR = "./pictures"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 16

MODALITY_LABELS = {
    "seq_only": "Seq Only",
    "bimodal": "Seq + SMILES",
    "trimodal": "Seq + SMILES + Struct",
}

MODALITY_COLORS = {
    "seq_only": "#1f77b4",
    "bimodal": "#ff7f0e",
    "trimodal": "#2ca02c",
}

DATA_ROUTING = {
    "kcat": {
        "master": "datasets/processed/kcat_max_wt_singleSeqs_wpdbs.csv",
        "test": "datasets/splits/kcat-seq_test_sequence_40cluster.csv",
    },
    "km": {
        "master": "datasets/processed/km_mean_wt_singleSeqs_wpdbs.csv",
        "test": "datasets/splits/km-seq_test_sequence_40cluster.csv",
    },
    "ki": {
        "master": "datasets/processed/ki_mean_wt_singleSeqs_wpdbs.csv",
        "test": "datasets/splits/ki-seq_test_sequence_40cluster.csv",
    },
}


def evaluate_single_model(task, mode, device, model=None):
    """Run OOD inference for one task/modality checkpoint."""
    cfg = DATA_ROUTING[task]
    weights_path = os.path.join(CHECKPOINT_DIR, f"best_model_{task}_{mode}.pt")

    if not os.path.exists(weights_path):
        print(f"[WARNING] Missing checkpoint: {weights_path}")
        return None

    test_dataset = EnzymeSubstrateDataset(cfg["master"], cfg["test"])
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    if model is None:
        model = HeteroscedasticEnzymeModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch_seqs, batch_smiles, batch_paths, batch_targets in tqdm(
            test_loader,
            desc=f"{task.upper()}-{mode}",
            leave=False,
        ):
            batch_targets = batch_targets.to(device)
            means, _ = model(batch_seqs, batch_smiles, batch_paths, device, modality=mode)
            all_targets.extend(batch_targets.cpu().numpy())
            all_preds.extend(means.cpu().numpy())

    metrics = compute_regression_metrics(all_targets, all_preds)
    return {
        "metrics": metrics,
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "n_samples": len(all_targets),
    }


def build_metrics_dataframe(results):
    """Flatten nested results into a tidy DataFrame."""
    rows = []
    for task in TASKS:
        for mode in MODALITIES:
            key = (task, mode)
            if key not in results:
                continue
            m = results[key]["metrics"]
            rows.append(
                {
                    "Task": task.upper(),
                    "Modality": MODALITY_LABELS[mode],
                    "ModeKey": mode,
                    "R2": m["R2"],
                    "Pearson": m["Pearson"],
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                }
            )
    return pd.DataFrame(rows)


def build_metric_matrix(df, metric_name):
    """Pivot metric values into task × modality matrix."""
    matrix = np.full((len(TASKS), len(MODALITIES)), np.nan)
    for i, task in enumerate(TASKS):
        for j, mode in enumerate(MODALITIES):
            row = df[(df["Task"] == task.upper()) & (df["ModeKey"] == mode)]
            if not row.empty:
                matrix[i, j] = row.iloc[0][metric_name]
    return matrix


def save_figure(fig, filename):
    path = os.path.join(PICTURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_metric_heatmaps(df):
    """Generate 4 heatmaps for R2, Pearson, RMSE, MAE."""
    for metric in METRIC_NAMES:
        matrix = build_metric_matrix(df, metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="YlOrRd" if metric in ("RMSE", "MAE") else "YlGnBu")
        ax.set_xticks(range(len(MODALITIES)))
        ax.set_xticklabels([MODALITY_LABELS[m] for m in MODALITIES], rotation=20, ha="right")
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels([t.upper() for t in TASKS])
        ax.set_title(f"OOD Performance Heatmap — {metric}", fontweight="bold", pad=12)

        for i in range(len(TASKS)):
            for j in range(len(MODALITIES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_figure(fig, f"heatmap_{metric.lower()}.png")


def plot_grouped_bar_charts(df):
    """Generate grouped bar charts for each metric."""
    x = np.arange(len(TASKS))
    bar_width = 0.25

    for metric in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(10, 6))
        for k, mode in enumerate(MODALITIES):
            values = []
            for task in TASKS:
                row = df[(df["Task"] == task.upper()) & (df["ModeKey"] == mode)]
                values.append(row.iloc[0][metric] if not row.empty else 0)
            offset = (k - 1) * bar_width
            rects = ax.bar(
                x + offset,
                values,
                bar_width,
                label=MODALITY_LABELS[mode],
                color=MODALITY_COLORS[mode],
                edgecolor="black",
                alpha=0.9,
            )
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(f"Modality Ablation — {metric} on OOD Test Set", fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([t.upper() for t in TASKS], fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        plt.tight_layout()
        save_figure(fig, f"bar_{metric.lower()}.png")


def plot_modality_trend_lines(df):
    """Line plots showing modality progression per kinetic parameter."""
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ["R2", "Pearson"]:
            values = []
            for mode in MODALITIES:
                row = df[(df["Task"] == task.upper()) & (df["ModeKey"] == mode)]
                values.append(row.iloc[0][metric] if not row.empty else np.nan)
            ax.plot(
                [MODALITY_LABELS[m] for m in MODALITIES],
                values,
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=metric,
            )
            for i, val in enumerate(values):
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.3f}",
                        (i, val),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=9,
                    )

        ax.set_title(f"Modality Scaling Trend — {task.upper()}", fontweight="bold", pad=12)
        ax.set_ylabel("Score", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        save_figure(fig, f"modality_trend_{task}.png")


def plot_error_trend_lines(df):
    """Line plots for RMSE/MAE across modalities per task."""
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ["RMSE", "MAE"]:
            values = []
            for mode in MODALITIES:
                row = df[(df["Task"] == task.upper()) & (df["ModeKey"] == mode)]
                values.append(row.iloc[0][metric] if not row.empty else np.nan)
            ax.plot(
                [MODALITY_LABELS[m] for m in MODALITIES],
                values,
                marker="s",
                linewidth=2.5,
                markersize=8,
                label=metric,
            )

        ax.set_title(f"Error Decay Trend — {task.upper()}", fontweight="bold", pad=12)
        ax.set_ylabel("Error (lower is better)", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        save_figure(fig, f"error_trend_{task}.png")


def plot_radar_charts(df):
    """One radar chart per kinetic parameter."""
    categories = ["R2", "Pearson", "RMSE", "MAE"]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    for task in TASKS:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="grey", size=9)

        for mode in MODALITIES:
            row = df[(df["Task"] == task.upper()) & (df["ModeKey"] == mode)]
            if row.empty:
                continue
            r = row.iloc[0]
            # Normalize RMSE/MAE to [0,1] scale for radar (invert so higher = better)
            rmse_norm = max(0, 1 - r["RMSE"] / 3.0)
            mae_norm = max(0, 1 - r["MAE"] / 3.0)
            values = [r["R2"], r["Pearson"], rmse_norm, mae_norm]
            values_closed = values + values[:1]
            ax.plot(
                angles,
                values_closed,
                linewidth=2,
                label=MODALITY_LABELS[mode],
                color=MODALITY_COLORS[mode],
            )
            ax.fill(angles, values_closed, color=MODALITY_COLORS[mode], alpha=0.08)

        ax.set_title(f"Performance Radar — {task.upper()}", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        plt.tight_layout()
        save_figure(fig, f"radar_{task}.png")


def plot_scatter_grid(results):
    """3×3 scatter grid: predicted vs actual for all 9 models."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for i, task in enumerate(TASKS):
        for j, mode in enumerate(MODALITIES):
            ax = axes[i, j]
            key = (task, mode)
            if key not in results:
                ax.set_visible(False)
                continue
            targets = results[key]["targets"]
            preds = results[key]["preds"]
            r2 = results[key]["metrics"]["R2"]

            ax.scatter(targets, preds, alpha=0.35, s=12, color=MODALITY_COLORS[mode], edgecolors="none")
            lims = [
                min(targets.min(), preds.min()),
                max(targets.max(), preds.max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"{task.upper()} | {MODALITY_LABELS[mode]}\nR2={r2:.3f}", fontsize=10)
            if i == 2:
                ax.set_xlabel("Actual (log scale)")
            if j == 0:
                ax.set_ylabel("Predicted (log scale)")

    fig.suptitle("Predicted vs Actual — All 9 Model Configurations", fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    save_figure(fig, "scatter_grid_all_models.png")


def plot_individual_scatter(results, task, mode):
    """High-resolution scatter for a single model."""
    key = (task, mode)
    if key not in results:
        return
    targets = results[key]["targets"]
    preds = results[key]["preds"]
    metrics = results[key]["metrics"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.4, s=18, color=MODALITY_COLORS[mode], edgecolors="none")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Ideal (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Value (log scale)", fontweight="bold")
    ax.set_ylabel("Predicted Value (log scale)", fontweight="bold")
    ax.set_title(
        f"{task.upper()} — {MODALITY_LABELS[mode]}\n"
        f"R2={metrics['R2']:.3f} | Pearson={metrics['Pearson']:.3f} | RMSE={metrics['RMSE']:.3f}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"scatter_{task}_{mode}.png")


def plot_residual_grid(results):
    """3×3 residual distribution grid."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for i, task in enumerate(TASKS):
        for j, mode in enumerate(MODALITIES):
            ax = axes[i, j]
            key = (task, mode)
            if key not in results:
                ax.set_visible(False)
                continue
            residuals = results[key]["preds"] - results[key]["targets"]
            ax.hist(residuals, bins=30, color=MODALITY_COLORS[mode], edgecolor="black", alpha=0.75)
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{task.upper()} | {MODALITY_LABELS[mode]}", fontsize=10)
            if i == 2:
                ax.set_xlabel("Residual (pred − actual)")
            if j == 0:
                ax.set_ylabel("Count")

    fig.suptitle("Residual Distributions — All 9 Models", fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    save_figure(fig, "residual_grid_all_models.png")


def plot_modality_gain(df):
    """Bar chart showing R² gain from seq_only baseline per task."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(TASKS))
    width = 0.35

    bimodal_gains, trimodal_gains = [], []
    for task in TASKS:
        base = df[(df["Task"] == task.upper()) & (df["ModeKey"] == "seq_only")]
        bi = df[(df["Task"] == task.upper()) & (df["ModeKey"] == "bimodal")]
        tri = df[(df["Task"] == task.upper()) & (df["ModeKey"] == "trimodal")]
        base_r2 = base.iloc[0]["R2"] if not base.empty else 0
        bimodal_gains.append(bi.iloc[0]["R2"] - base_r2 if not bi.empty else 0)
        trimodal_gains.append(tri.iloc[0]["R2"] - base_r2 if not tri.empty else 0)

    ax.bar(x - width / 2, bimodal_gains, width, label="dR2 (Bimodal - Seq Only)", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width / 2, trimodal_gains, width, label="dR2 (Trimodal - Seq Only)", color="#2ca02c", edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in TASKS], fontweight="bold")
    ax.set_ylabel("dR2 vs Seq-Only Baseline", fontweight="bold")
    ax.set_title("Multimodal Fusion Gain Analysis", fontweight="bold", pad=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure(fig, "modality_gain_delta_r2.png")


def plot_best_model_comparison(df):
    """Compare best modality per task side-by-side."""
    fig, ax = plt.subplots(figsize=(10, 6))
    best_rows = []
    for task in TASKS:
        sub = df[df["Task"] == task.upper()]
        if sub.empty:
            continue
        best = sub.loc[sub["R2"].idxmax()]
        best_rows.append(best)

    if not best_rows:
        plt.close(fig)
        return

    best_df = pd.DataFrame(best_rows)
    x = np.arange(len(best_df))
    width = 0.2
    for k, metric in enumerate(METRIC_NAMES):
        offset = (k - 1.5) * width
        vals = best_df[metric].values
        if metric in ("RMSE", "MAE"):
            vals = vals / vals.max() if vals.max() > 0 else vals
        ax.bar(x + offset, vals, width, label=metric, edgecolor="black", alpha=0.85)

    labels = [f"{r['Task']}\n({r['Modality']})" for _, r in best_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_title("Best Model per Task (Highest R2)", fontweight="bold", pad=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure(fig, "best_model_per_task.png")


def plot_summary_table(df):
    """Render metrics table as an image."""
    display_df = df[["Task", "Modality", "R2", "Pearson", "RMSE", "MAE"]].copy()
    for col in ["R2", "Pearson", "RMSE", "MAE"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(display_df) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    ax.set_title("Full OOD Evaluation Summary (9 Models)", fontweight="bold", pad=20, fontsize=13)
    plt.tight_layout()
    save_figure(fig, "summary_table.png")


def generate_all_visualizations(results, df):
    """Orchestrate all plot generation (~18 figures)."""
    print("\n[Visualization] Generating plots...")
    plot_metric_heatmaps(df)           # 4 images
    plot_grouped_bar_charts(df)        # 4 images
    plot_modality_trend_lines(df)      # 3 images
    plot_error_trend_lines(df)         # 3 images
    plot_radar_charts(df)              # 3 images
    plot_scatter_grid(results)         # 1 image
    plot_residual_grid(results)        # 1 image
    plot_modality_gain(df)             # 1 image
    plot_best_model_comparison(df)     # 1 image
    plot_summary_table(df)             # 1 image

    for task in TASKS:
        for mode in MODALITIES:
            plot_individual_scatter(results, task, mode)  # up to 9 images


def evaluate_pipeline():
    os.makedirs(PICTURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("  BATCH OOD EVALUATION — 9 MODEL MATRIX (3 Tasks × 3 Modalities)")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    results = {}
    print("[*] Loading shared encoder backbone (one-time)...")
    shared_model = HeteroscedasticEnzymeModel().to(device)

    for task in TASKS:
        for mode in MODALITIES:
            print(f"[*] Evaluating {task.upper()} | {MODALITY_LABELS[mode]} ...")
            out = evaluate_single_model(task, mode, device, model=shared_model)
            if out is not None:
                results[(task, mode)] = out
                m = out["metrics"]
                print(
                    f"    R2={m['R2']:.4f} | Pearson={m['Pearson']:.4f} | "
                    f"RMSE={m['RMSE']:.4f} | MAE={m['MAE']:.4f} | n={out['n_samples']}"
                )

    if not results:
        print("\n[ERROR] No checkpoints found. Train models first (python train.py).")
        return

    df = build_metrics_dataframe(results)

    print(f"\n{'=' * 60}")
    print("  FINAL OOD RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(df[["Task", "Modality", "R2", "Pearson", "RMSE", "MAE"]].to_string(index=False))
    print(f"{'=' * 60}")

    generate_all_visualizations(results, df)

    n_plots = len([f for f in os.listdir(PICTURES_DIR) if f.endswith(".png")])
    print(f"\n[SUCCESS] Evaluation complete. {len(results)}/9 models evaluated.")
    print(f"[SUCCESS] {n_plots} plots saved to '{PICTURES_DIR}/'")


if __name__ == "__main__":
    evaluate_pipeline()
