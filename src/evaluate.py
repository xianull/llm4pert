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
  - Breakdown by subgroup: unseen_single, combo_seen0/1/2
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from collections import defaultdict
from typing import Dict, List, Optional
from torch.utils.data import DataLoader


METRIC_NAMES = [
    "mse",
    "mae",
    "pearson_top20",
    "pearson_delta_top20",
    "direction_accuracy",
]


def _compute_pert_metrics(
    mean_pred: np.ndarray,
    mean_truth: np.ndarray,
    mean_ctrl: np.ndarray,
    de_idx: np.ndarray,
    de_len: int,
) -> Dict[str, float]:
    """Compute all metrics for a single perturbation (already cell-averaged)."""
    mse = float(np.mean((mean_pred - mean_truth) ** 2))

    de_genes = de_idx[:de_len]
    pred_de = mean_pred[de_genes]
    truth_de = mean_truth[de_genes]
    pred_delta = pred_de - mean_ctrl[de_genes]
    true_delta = truth_de - mean_ctrl[de_genes]

    mae_delta = float(np.mean(np.abs(pred_delta - true_delta)))

    if len(pred_de) >= 2 and np.std(pred_de) > 0 and np.std(truth_de) > 0:
        r, _ = pearsonr(pred_de, truth_de)
    else:
        r = 0.0

    if len(pred_delta) >= 2 and np.std(pred_delta) > 0 and np.std(true_delta) > 0:
        r_delta, _ = pearsonr(pred_delta, true_delta)
    else:
        r_delta = 0.0

    dir_match = np.sign(pred_delta) == np.sign(true_delta)
    dir_acc = float(np.mean(dir_match))

    return {
        "mse": mse,
        "mae": mae_delta,
        "pearson_top20": r,
        "pearson_delta_top20": r_delta,
        "direction_accuracy": dir_acc,
    }


def _classify_perturbation(
    pert_name: str,
    subgroup: Optional[Dict[str, str]] = None,
    train_pert_genes: Optional[set] = None,
) -> str:
    """Classify a perturbation into a subgroup.

    Uses GEARS subgroup dict if available, otherwise infers from name
    and training gene set.

    Categories:
      - unseen_single:  single gene, not seen in training
      - seen_single:    single gene, seen in training combos
      - combo_seen0:    two genes, neither seen
      - combo_seen1:    two genes, one seen
      - combo_seen2:    two genes, both seen
    """
    if subgroup is not None and pert_name in subgroup:
        return subgroup[pert_name]

    parts = [g for g in pert_name.split("+") if g != "ctrl"]

    if len(parts) <= 1:
        return "unseen_single"

    if train_pert_genes is not None:
        seen_count = sum(1 for g in parts if g in train_pert_genes)
        return f"combo_seen{seen_count}"

    return "combo"


def evaluate_model(
    model,
    dataset,
    device: torch.device,
    cfg,
    collate_fn,
    subgroup: Optional[Dict[str, str]] = None,
    train_pert_genes: Optional[set] = None,
) -> Dict[str, float]:
    """Evaluate DyGenePT on a dataset split.

    Args:
        model:            Trained model.
        dataset:          PerturbationDataset for evaluation split.
        device:           Torch device.
        cfg:              Config.
        collate_fn:       Batch collation function.
        subgroup:         Optional dict mapping pert_name -> subgroup string
                          (from GEARS pert_data.subgroup).
        train_pert_genes: Optional set of gene symbols seen in training perts,
                          used for combo_seen classification when subgroup is absent.

    Returns:
        Dict with overall metrics and per-subgroup breakdown.
    """
    model.eval()

    eval_batch_size = cfg.training.batch_size * 4

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

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            gene_ids = batch["gene_ids"].to(device, non_blocking=True)
            values = batch["values"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            pert_gene_indices = batch["pert_gene_indices"].to(device, non_blocking=True)
            ctrl_expression = batch["ctrl_expression"].to(device, non_blocking=True)
            pert_type_ids = batch["pert_type_ids"].to(device, non_blocking=True) if "pert_type_ids" in batch else None

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(
                    gene_ids=gene_ids,
                    values=values,
                    padding_mask=padding_mask,
                    pert_gene_indices=pert_gene_indices,
                    ctrl_expression=ctrl_expression,
                    pert_type_ids=pert_type_ids,
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

    # Compute per-perturbation metrics and group by subgroup
    all_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for pert_name, data in results_per_pert.items():
        de_idx = data["de_idx"]
        de_len = data["de_idx_len"]
        if de_len == 0:
            continue

        mean_pred = np.mean(np.stack(data["preds"]), axis=0)
        mean_truth = np.mean(np.stack(data["truths"]), axis=0)
        mean_ctrl = np.mean(np.stack(data["ctrls"]), axis=0)

        pert_metrics = _compute_pert_metrics(
            mean_pred, mean_truth, mean_ctrl, de_idx, de_len
        )

        group = _classify_perturbation(pert_name, subgroup, train_pert_genes)
        all_metrics[group].append(pert_metrics)
        all_metrics["_all"].append(pert_metrics)

    # Aggregate: mean per subgroup
    result = {}

    # Overall metrics
    for metric in METRIC_NAMES:
        vals = [m[metric] for m in all_metrics["_all"]]
        result[metric] = float(np.mean(vals)) if vals else 0.0

    # Per-subgroup metrics
    for group in sorted(all_metrics.keys()):
        if group == "_all":
            continue
        n = len(all_metrics[group])
        result[f"_n_{group}"] = n
        for metric in METRIC_NAMES:
            vals = [m[metric] for m in all_metrics[group]]
            result[f"{group}/{metric}"] = float(np.mean(vals)) if vals else 0.0

    model.train()
    return result


def format_subgroup_table(metrics: Dict[str, float]) -> str:
    """Format per-subgroup evaluation results as a readable table.

    Args:
        metrics: Dict from evaluate_model (contains group/metric keys).

    Returns:
        Formatted string table.
    """
    # Discover subgroups
    groups = sorted(set(
        k.rsplit("/", 1)[0] for k in metrics if "/" in k and not k.startswith("_")
    ))

    if not groups:
        return ""

    lines = []
    lines.append("=" * 90)
    lines.append("Per-Subgroup Evaluation Breakdown")
    lines.append("=" * 90)

    header = f"{'Subgroup':<20} {'N':>4}"
    for m in ["pearson_delta_top20", "pearson_top20", "mse", "mae", "direction_accuracy"]:
        short = m.replace("pearson_delta_top20", "P_delta").replace(
            "pearson_top20", "P_abs").replace(
            "direction_accuracy", "Dir_acc")
        header += f" {short:>10}"
    lines.append(header)
    lines.append("-" * 90)

    for group in groups:
        n = int(metrics.get(f"_n_{group}", 0))
        row = f"{group:<20} {n:>4}"
        for m in ["pearson_delta_top20", "pearson_top20", "mse", "mae", "direction_accuracy"]:
            val = metrics.get(f"{group}/{m}", float("nan"))
            row += f" {val:>10.4f}"
        lines.append(row)

    # Overall
    n_total = sum(int(metrics.get(f"_n_{g}", 0)) for g in groups)
    row = f"{'OVERALL':<20} {n_total:>4}"
    for m in ["pearson_delta_top20", "pearson_top20", "mse", "mae", "direction_accuracy"]:
        val = metrics.get(m, float("nan"))
        row += f" {val:>10.4f}"
    lines.append("-" * 90)
    lines.append(row)
    lines.append("=" * 90)

    return "\n".join(lines)


def format_comparison_table(
    dygenept_metrics: Dict[str, float],
    dataset_name: str,
) -> str:
    """Format metrics as a comparison table against LangPert baselines."""
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
