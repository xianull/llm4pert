"""Evaluation metrics for DyGenePT.

Computes per-perturbation MSE, MAE, Pearson correlation on top DE genes,
and direction accuracy — matching both GEARS and LangPert evaluation protocols.

LangPert protocol (Replogle K562/RPE1):
  - Metrics computed on delta (pred - ctrl) for top-20 DE genes
  - Pearson correlation, MAE, MSE on delta vectors
  - 5-fold cross-validation, averaged across folds and perturbations

GEARS protocol (Norman):
  - Metrics computed per-perturbation, averaged
  - Pearson on absolute expression and on delta
  - Direction accuracy on top-20 DE genes
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from collections import defaultdict
from typing import Dict, Optional
from torch.utils.data import DataLoader


def evaluate_model(
    model,
    dataset,
    device: torch.device,
    cfg,
    collate_fn,
) -> Dict[str, float]:
    """Evaluate DyGenePT on a dataset split.

    Metrics:
      1. mse:                 MSE across all genes, averaged per perturbation
      2. mae:                 MAE on delta for top-20 DE genes (LangPert metric)
      3. pearson_top20:       Pearson r on top-20 DE genes (absolute expression)
      4. pearson_delta_top20: Pearson r on delta (pred−ctrl vs true−ctrl) for top-20 DE
      5. direction_accuracy:  Fraction of top-20 DE genes where predicted direction
                              of change matches true direction

    All metrics are computed per-perturbation, then averaged.

    Returns:
        Dict with metric names and their mean values.
    """
    model.eval()

    eval_batch_size = cfg.training.batch_size * 2

    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Collect predictions grouped by perturbation name
    results_per_pert: Dict[str, dict] = defaultdict(
        lambda: {"preds": [], "truths": [], "ctrls": [], "de_idx": None, "de_idx_len": 0}
    )

    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            pert_gene_indices = batch["pert_gene_indices"].to(device)
            ctrl_expression = batch["ctrl_expression"].to(device)

            output = model(
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                pert_gene_indices=pert_gene_indices,
                ctrl_expression=ctrl_expression,
            )

            pred = output["pred_expression"].cpu().numpy()
            target = batch["target_expression"].numpy()
            ctrl = batch["ctrl_expression"].numpy()
            de_idx = batch["de_idx"].numpy()
            de_idx_len = batch["de_idx_len"].numpy()
            pert_names = batch["pert_names"]

            for i, pname in enumerate(pert_names):
                results_per_pert[pname]["preds"].append(pred[i])
                results_per_pert[pname]["truths"].append(target[i])
                results_per_pert[pname]["ctrls"].append(ctrl[i])
                if results_per_pert[pname]["de_idx"] is None:
                    results_per_pert[pname]["de_idx"] = de_idx[i]
                    results_per_pert[pname]["de_idx_len"] = int(de_idx_len[i])

    # Compute per-perturbation metrics
    metrics = {
        "mse": [],
        "mae": [],
        "pearson_top20": [],
        "pearson_delta_top20": [],
        "direction_accuracy": [],
    }

    for pert_name, data in results_per_pert.items():
        preds = np.stack(data["preds"])
        truths = np.stack(data["truths"])
        ctrls = np.stack(data["ctrls"])
        de_idx = data["de_idx"]
        de_len = data["de_idx_len"]

        if de_len == 0:
            continue

        # Average across cells sharing the same perturbation
        mean_pred = np.mean(preds, axis=0)
        mean_truth = np.mean(truths, axis=0)
        mean_ctrl = np.mean(ctrls, axis=0)

        # MSE (all genes)
        mse = float(np.mean((mean_pred - mean_truth) ** 2))
        metrics["mse"].append(mse)

        # Top DE genes
        de_genes = de_idx[:de_len]

        # Delta vectors (for LangPert-compatible metrics)
        pred_de = mean_pred[de_genes]
        truth_de = mean_truth[de_genes]
        pred_delta = pred_de - mean_ctrl[de_genes]
        true_delta = truth_de - mean_ctrl[de_genes]

        # MAE on delta (LangPert metric)
        mae_delta = float(np.mean(np.abs(pred_delta - true_delta)))
        metrics["mae"].append(mae_delta)

        # Pearson on absolute expression (top DE)
        if len(pred_de) >= 2 and np.std(pred_de) > 0 and np.std(truth_de) > 0:
            r, _ = pearsonr(pred_de, truth_de)
            metrics["pearson_top20"].append(r)
        else:
            metrics["pearson_top20"].append(0.0)

        # Pearson on delta (pred-ctrl vs truth-ctrl)
        if len(pred_delta) >= 2 and np.std(pred_delta) > 0 and np.std(true_delta) > 0:
            r_delta, _ = pearsonr(pred_delta, true_delta)
            metrics["pearson_delta_top20"].append(r_delta)
        else:
            metrics["pearson_delta_top20"].append(0.0)

        # Direction accuracy
        dir_match = np.sign(pred_delta) == np.sign(true_delta)
        metrics["direction_accuracy"].append(float(np.mean(dir_match)))

    # Average across perturbations
    result = {}
    for k, v in metrics.items():
        result[k] = float(np.mean(v)) if v else 0.0

    model.train()
    return result


def format_comparison_table(
    dygenept_metrics: Dict[str, float],
    dataset_name: str,
) -> str:
    """Format metrics as a comparison table against LangPert baselines.

    LangPert reported results (from paper Table 1, Simulation split):
      K562:  Corr=0.731, MAE=0.224, MSE=0.097
      RPE1:  Corr=0.772, MAE=0.318, MSE=0.187

    Args:
        dygenept_metrics: Dict of DyGenePT evaluation metrics.
        dataset_name: One of 'replogle_k562_essential' or 'replogle_rpe1_essential'.

    Returns:
        Formatted string table for logging.
    """
    langpert_baselines = {
        "replogle_k562_essential": {"pearson_delta_top20": 0.731, "mae": 0.224, "mse_de": 0.097},
        "replogle_rpe1_essential": {"pearson_delta_top20": 0.772, "mae": 0.318, "mse_de": 0.187},
    }

    lines = []
    lines.append("=" * 70)
    lines.append(f"Comparison with LangPert — {dataset_name}")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'DyGenePT':>12} {'LangPert':>12}")
    lines.append("-" * 70)

    baseline = langpert_baselines.get(dataset_name, {})

    for metric_name in ["pearson_delta_top20", "mae", "mse"]:
        ours = dygenept_metrics.get(metric_name, float("nan"))
        theirs = baseline.get(metric_name, "N/A")
        theirs_str = f"{theirs:.4f}" if isinstance(theirs, float) else str(theirs)
        lines.append(f"{metric_name:<30} {ours:>12.4f} {theirs_str:>12}")

    for metric_name in ["pearson_top20", "direction_accuracy"]:
        ours = dygenept_metrics.get(metric_name, float("nan"))
        lines.append(f"{metric_name:<30} {ours:>12.4f} {'N/A':>12}")

    lines.append("=" * 70)
    return "\n".join(lines)
