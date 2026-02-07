"""DyGenePT cross-validation training for LangPert-comparable evaluation.

Implements K-fold cross-validation on Replogle K562/RPE1 datasets to produce
results directly comparable to LangPert (ICLR 2025 MLGenX).

LangPert protocol:
  - 5-fold CV on single-gene perturbations
  - Metrics: Pearson correlation, MAE, MSE on delta (pred - ctrl) for top-20 DE genes
  - Results averaged across folds

Usage:
    python -m src.train_cv --config configs/replogle_k562.yaml
    python -m src.train_cv --config configs/replogle_rpe1.yaml
"""

import argparse
import json
from pathlib import Path
import numpy as np

import src.torchtext_shim  # noqa: F401 — must be before any scgpt import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.utils import set_seed, get_device, setup_logging
from src.model.dygenept import DyGenePT
from src.data.dataset import PerturbationDataset
from src.data.collator import collate_perturbation_batch
from src.evaluate import evaluate_model, format_comparison_table
from src.train import compute_loss, WarmupCosineScheduler


def get_perturbation_folds(dataset, n_folds: int, seed: int):
    """Split dataset indices into K folds based on perturbation identity.

    Ensures that all cells from the same perturbation are in the same fold,
    matching LangPert's per-perturbation split strategy.

    Args:
        dataset: PerturbationDataset instance.
        n_folds: Number of cross-validation folds.
        seed: Random seed for reproducible splits.

    Returns:
        List of (train_indices, test_indices) tuples for each fold.
    """
    # Group sample indices by perturbation name
    pert_to_indices = {}
    for i in range(len(dataset)):
        data = dataset.data_list[i]
        pert_name = data.pert if isinstance(data.pert, str) else data.pert[0]
        if pert_name not in pert_to_indices:
            pert_to_indices[pert_name] = []
        pert_to_indices[pert_name].append(i)

    # Shuffle perturbation names and split into folds
    rng = np.random.RandomState(seed)
    pert_names = sorted(pert_to_indices.keys())
    rng.shuffle(pert_names)

    fold_size = len(pert_names) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            test_perts = pert_names[start:]
        else:
            test_perts = pert_names[start : start + fold_size]
        train_perts = [p for p in pert_names if p not in set(test_perts)]

        test_indices = []
        for p in test_perts:
            test_indices.extend(pert_to_indices[p])

        train_indices = []
        for p in train_perts:
            train_indices.extend(pert_to_indices[p])

        folds.append((train_indices, test_indices))

    return folds


def train_one_fold(
    cfg,
    model,
    train_dataset,
    test_dataset,
    device,
    logger,
    fold_idx: int,
    output_dir: Path,
):
    """Train and evaluate a single CV fold.

    Args:
        cfg: Configuration object.
        model: Fresh DyGenePT model (re-initialized for each fold).
        train_dataset: Subset for this fold's training.
        test_dataset: Subset for this fold's testing.
        device: torch device.
        logger: Logger.
        fold_idx: Fold number (0-indexed).
        output_dir: Base output directory.

    Returns:
        Dict of test metrics for this fold.
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Split training data into train/val (90%/10%) for early stopping
    n_train = len(train_dataset)
    n_val = max(1, int(n_train * 0.1))
    n_train_actual = n_train - n_val
    rng = torch.Generator().manual_seed(cfg.training.seed + fold_idx)
    train_split, val_split = random_split(
        train_dataset, [n_train_actual, n_val], generator=rng,
    )

    train_loader = DataLoader(
        train_split,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_perturbation_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_split,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_perturbation_batch,
        pin_memory=True,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx + 1} Epoch {epoch + 1}/{cfg.training.epochs}",
        )
        for batch in pbar:
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            pert_gene_indices = batch["pert_gene_indices"].to(device)
            ctrl_expression = batch["ctrl_expression"].to(device)
            target_expression = batch["target_expression"].to(device)
            de_idx = batch["de_idx"].to(device)
            de_idx_len = batch["de_idx_len"].to(device)

            output = model(
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                pert_gene_indices=pert_gene_indices,
                ctrl_expression=ctrl_expression,
            )

            losses = compute_loss(
                pred=output["pred_expression"],
                target=target_expression,
                de_idx=de_idx,
                de_idx_len=de_idx_len,
                ctrl_expression=ctrl_expression,
                mse_weight=cfg.training.loss.mse_weight,
                direction_weight=cfg.training.loss.direction_weight,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += losses["total"].item()
            num_batches += 1

            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)

        # Compute validation loss for early stopping
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                gene_ids = batch["gene_ids"].to(device)
                values = batch["values"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                pert_gene_indices = batch["pert_gene_indices"].to(device)
                ctrl_expression = batch["ctrl_expression"].to(device)
                target_expression = batch["target_expression"].to(device)
                de_idx = batch["de_idx"].to(device)
                de_idx_len = batch["de_idx_len"].to(device)

                output = model(
                    gene_ids=gene_ids,
                    values=values,
                    padding_mask=padding_mask,
                    pert_gene_indices=pert_gene_indices,
                    ctrl_expression=ctrl_expression,
                )
                losses_val = compute_loss(
                    pred=output["pred_expression"],
                    target=target_expression,
                    de_idx=de_idx,
                    de_idx_len=de_idx_len,
                    ctrl_expression=ctrl_expression,
                    mse_weight=cfg.training.loss.mse_weight,
                    direction_weight=cfg.training.loss.direction_weight,
                )
                val_loss += losses_val["total"].item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        logger.info(
            f"Fold {fold_idx + 1}, Epoch {epoch + 1}: "
            f"train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(f"Fold {fold_idx + 1}: early stopping at epoch {epoch + 1}")
                break

    # Load best model and evaluate on test fold
    model.load_state_dict(
        torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_metrics = evaluate_model(
        model, test_dataset, device, cfg, collate_perturbation_batch
    )

    logger.info(f"Fold {fold_idx + 1} test: " + ", ".join(
        f"{k}={v:.4f}" for k, v in test_metrics.items()
    ))

    with open(fold_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def train_cv(cfg):
    """Run K-fold cross-validation training and evaluation."""
    logger = setup_logging(cfg.paths.output_dir)
    set_seed(cfg.training.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    n_folds = cfg.training.cross_validation.n_folds
    dataset_name = cfg.training.dataset

    # ------------------------------------------------------------------
    # 1. Load GEARS data (full dataset, we handle splits ourselves)
    # ------------------------------------------------------------------
    from gears import PertData

    if hasattr(cfg, "dataset_configs") and dataset_name in cfg.dataset_configs:
        ds_cfg = cfg.dataset_configs[dataset_name]
        data_name = ds_cfg.data_name
    else:
        data_name = dataset_name

    logger.info(f"Loading GEARS dataset: {data_name}")
    pert_data = PertData(cfg.paths.perturb_data_dir)
    pert_data.load(data_name=data_name)
    # Use simulation split to get GEARS to prepare the data properly,
    # then we override with our own CV folds
    pert_data.prepare_split(split="simulation", seed=cfg.training.seed)

    # Gene names
    if "gene_name" in pert_data.adata.var.columns:
        gene_names = list(pert_data.adata.var["gene_name"])
    else:
        gene_names = list(pert_data.adata.var_names)
    num_genes = len(gene_names)
    logger.info(f"Number of genes: {num_genes}")

    # ------------------------------------------------------------------
    # 2. Load precomputed facets
    # ------------------------------------------------------------------
    saved = torch.load(cfg.paths.facet_embeddings, map_location="cpu", weights_only=False)
    gene_to_facet_idx = saved["gene_to_idx"]
    logger.info(f"Loaded facet tensor with {len(gene_to_facet_idx)} genes")

    # ------------------------------------------------------------------
    # 3. Build full dataset (train+val+test combined for CV)
    # ------------------------------------------------------------------
    # Load scGPT vocab directly without instantiating full model
    from scgpt.tokenizer import GeneVocab
    ckpt_dir = Path(cfg.paths.scgpt_checkpoint)
    scgpt_vocab = GeneVocab.from_file(ckpt_dir / "vocab.json")

    # Combine all splits into one dataset for CV
    full_datasets = []
    for split_name in ["train", "val", "test"]:
        try:
            ds = PerturbationDataset(
                pert_data, split_name, gene_to_facet_idx, scgpt_vocab, gene_names,
                max_seq_len=cfg.cell_encoder.max_seq_len,
            )
            full_datasets.append(ds)
        except (KeyError, RuntimeError):
            logger.info(f"Split '{split_name}' not available, skipping")

    # Merge data_lists from all splits
    merged_data_list = []
    for ds in full_datasets:
        merged_data_list.extend(ds.data_list)

    # Create a single dataset with the merged data
    full_dataset = full_datasets[0]  # re-use first for structure
    full_dataset.data_list = merged_data_list
    logger.info(f"Total samples for CV: {len(full_dataset)}")

    # ------------------------------------------------------------------
    # 4. Generate CV folds
    # ------------------------------------------------------------------
    folds = get_perturbation_folds(full_dataset, n_folds, cfg.training.seed)
    logger.info(f"Generated {n_folds} folds")
    for i, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"  Fold {i + 1}: train={len(train_idx)}, test={len(test_idx)}")

    # ------------------------------------------------------------------
    # 5. Train each fold
    # ------------------------------------------------------------------
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Fold {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        set_seed(cfg.training.seed + fold_idx)

        # Re-initialize model for each fold
        model = DyGenePT(cfg, num_genes=num_genes)
        model = model.to(device)

        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        fold_metrics = train_one_fold(
            cfg, model, train_subset, test_subset,
            device, logger, fold_idx, output_dir,
        )
        all_fold_metrics.append(fold_metrics)

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # 6. Aggregate CV results
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")

    cv_summary = {}
    metric_names = all_fold_metrics[0].keys()

    for metric in metric_names:
        values = [m[metric] for m in all_fold_metrics]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        cv_summary[metric] = {"mean": mean_val, "std": std_val}
        logger.info(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f}")

    # LangPert comparison
    mean_metrics = {k: v["mean"] for k, v in cv_summary.items()}
    comparison = format_comparison_table(mean_metrics, dataset_name)
    logger.info("\n" + comparison)

    # Save
    cv_results = {
        "n_folds": n_folds,
        "dataset": dataset_name,
        "per_fold": all_fold_metrics,
        "summary": cv_summary,
    }
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    logger.info(f"CV results saved to {output_dir / 'cv_results.json'}")
    return cv_results


def main():
    parser = argparse.ArgumentParser(
        description="DyGenePT Cross-Validation Training"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cv(cfg)


if __name__ == "__main__":
    main()
