"""DyGenePT training loop with DDP multi-GPU support.

Usage:
    Single GPU:  python -m src.train --config configs/default.yaml
    Multi GPU:   torchrun --nproc_per_node=8 -m src.train --config configs/default.yaml
"""

import argparse
import math
import os
from pathlib import Path

import src.torchtext_shim  # noqa: F401 — must be before any scgpt import

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.utils import set_seed, get_device, setup_logging
from src.model.dygenept import DyGenePT
from src.data.dataset import PerturbationDataset
from src.data.collator import collate_perturbation_batch
from src.evaluate import evaluate_model, format_comparison_table


# =====================================================================
# DDP helpers
# =====================================================================
def setup_ddp():
    """Initialize distributed process group. Returns (rank, local_rank, world_size).
    If not launched via torchrun, returns (0, 0, 1) for single-GPU fallback.
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# =====================================================================
# Loss function
# =====================================================================
def compute_loss(
    pred: torch.Tensor,          # (B, G)
    target: torch.Tensor,        # (B, G)
    de_idx: torch.LongTensor,   # (B, top_de)
    de_idx_len: torch.LongTensor,  # (B,)
    ctrl_expression: torch.Tensor,  # (B, G)
    mse_weight: float = 1.0,
    de_mse_weight: float = 0.5,
    direction_weight: float = 0.1,
) -> dict:
    """Combined MSE + DE-focused MSE + direction loss.

    MSE loss:       on all genes.
    DE MSE loss:    focused MSE on top DE genes only.
    Direction loss:  on top DE genes, ensuring predicted direction of change
                     (up/down relative to control) matches ground truth.
    """
    # Global MSE
    loss_mse = nn.functional.mse_loss(pred, target)

    # Direction loss and DE-focused MSE on DE genes
    if (de_mse_weight > 0 or direction_weight > 0) and de_idx_len.sum() > 0:
        # Clamp de_idx to valid range to prevent OOB on gather
        de_idx = de_idx.clamp(0, pred.shape[1] - 1)

        # Gather DE gene predictions and targets
        # de_idx: (B, top_de) -> use as gather indices
        pred_de = pred.gather(1, de_idx)      # (B, top_de)
        target_de = target.gather(1, de_idx)  # (B, top_de)
        ctrl_de = ctrl_expression.gather(1, de_idx)

        # Mask out padded DE indices
        mask = torch.arange(de_idx.shape[1], device=de_idx.device).unsqueeze(0)
        mask = mask < de_idx_len.unsqueeze(1)  # (B, top_de) bool

        # DE-focused MSE (masked)
        if de_mse_weight > 0:
            de_mse_raw = nn.functional.mse_loss(pred_de, target_de, reduction="none")
            loss_de_mse = (de_mse_raw * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            loss_de_mse = torch.tensor(0.0, device=pred.device)

        # Direction loss
        if direction_weight > 0:
            pred_delta = pred_de - ctrl_de
            target_delta = target_de - ctrl_de
            target_dir = (target_delta > 0).float()

            loss_dir = nn.functional.binary_cross_entropy_with_logits(
                pred_delta, target_dir, reduction="none"
            )
            loss_dir = (loss_dir * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            loss_dir = torch.tensor(0.0, device=pred.device)
    else:
        loss_de_mse = torch.tensor(0.0, device=pred.device)
        loss_dir = torch.tensor(0.0, device=pred.device)

    total = mse_weight * loss_mse + de_mse_weight * loss_de_mse + direction_weight * loss_dir
    return {"total": total, "mse": loss_mse, "de_mse": loss_de_mse, "direction": loss_dir}


# =====================================================================
# Warmup + Cosine scheduler
# =====================================================================
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(
                self.min_lr / self.base_lrs[0],
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# =====================================================================
# Main training function
# =====================================================================
def train(cfg):
    # --- DDP setup ---
    rank, local_rank, world_size = setup_ddp()
    distributed = world_size > 1

    logger = setup_logging(cfg.paths.output_dir)
    set_seed(cfg.training.seed + rank)  # different seed per rank for data diversity

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device()

    if is_main_process(rank):
        logger.info(f"Device: {device}, world_size: {world_size}")

    # ------------------------------------------------------------------
    # 1. Load GEARS data
    # ------------------------------------------------------------------
    from gears import PertData

    dataset_name = cfg.training.dataset
    # Use dataset-specific config if available, otherwise use defaults
    if hasattr(cfg, "dataset_configs") and dataset_name in cfg.dataset_configs:
        ds_cfg = cfg.dataset_configs[dataset_name]
        data_name = ds_cfg.data_name
        split_mode = ds_cfg.get("split_mode", "simulation")
    else:
        data_name = dataset_name
        split_mode = "simulation"

    if is_main_process(rank):
        logger.info(f"Loading GEARS dataset: {data_name} (split_mode={split_mode})")
    pert_data = PertData(cfg.paths.perturb_data_dir)
    local_path = os.path.join(cfg.paths.perturb_data_dir, data_name)
    if os.path.exists(local_path):
        pert_data.load(data_path=local_path)
    else:
        pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split_mode, seed=cfg.training.seed)

    # Gene names list
    if "gene_name" in pert_data.adata.var.columns:
        gene_names = list(pert_data.adata.var["gene_name"])
    else:
        gene_names = list(pert_data.adata.var_names)
    num_genes = len(gene_names)
    if is_main_process(rank):
        logger.info(f"Number of genes: {num_genes}")

    # ------------------------------------------------------------------
    # 2. Load precomputed gene facet data
    # ------------------------------------------------------------------
    saved = torch.load(cfg.paths.facet_embeddings, map_location="cpu", weights_only=False)
    gene_to_facet_idx = saved["gene_to_idx"]
    if is_main_process(rank):
        logger.info(f"Loaded facet tensor with {len(gene_to_facet_idx)} genes")

    # ------------------------------------------------------------------
    # 3. Initialize model
    # ------------------------------------------------------------------
    if is_main_process(rank):
        logger.info("Initializing DyGenePT model...")
    model = DyGenePT(cfg, num_genes=num_genes)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Unwrap helper: access underlying module for saving/loading/eval
    raw_model = model.module if distributed else model

    # ------------------------------------------------------------------
    # 4. Build datasets
    # ------------------------------------------------------------------
    scgpt_vocab = raw_model.cell_encoder.vocab

    if is_main_process(rank):
        print("Creating dataloaders....")
    train_dataset = PerturbationDataset(
        pert_data, "train", gene_to_facet_idx, scgpt_vocab, gene_names,
        max_seq_len=cfg.cell_encoder.max_seq_len,
    )
    val_dataset = PerturbationDataset(
        pert_data, "val", gene_to_facet_idx, scgpt_vocab, gene_names,
        max_seq_len=cfg.cell_encoder.max_seq_len,
    )
    test_dataset = PerturbationDataset(
        pert_data, "test", gene_to_facet_idx, scgpt_vocab, gene_names,
        max_seq_len=cfg.cell_encoder.max_seq_len,
    )

    # Use DistributedSampler for training data in DDP
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_perturbation_batch,
        pin_memory=True,
    )
    if is_main_process(rank):
        print("Done!")

    # ------------------------------------------------------------------
    # 5. Optimizer and scheduler
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Optional: wandb logging (rank 0 only)
    # ------------------------------------------------------------------
    use_wandb = False
    if is_main_process(rank):
        try:
            import wandb

            wandb.init(project="DyGenePT", config=dict(cfg))
            use_wandb = True
            logger.info("Weights & Biases logging enabled")
        except Exception:
            logger.info("Weights & Biases not available, using local logging only")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    best_val_mse = float("inf")
    patience_counter = 0
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process(rank):
        logger.info(
            f"Starting training: {cfg.training.epochs} epochs, "
            f"batch_size={cfg.training.batch_size}x{world_size}gpu, lr={cfg.training.lr}"
        )

    for epoch in range(cfg.training.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_de_mse = 0.0
        epoch_dir = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.epochs}",
            disable=not is_main_process(rank),
        )
        for batch in pbar:
            # Move to device
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            pert_gene_indices = batch["pert_gene_indices"].to(device)
            ctrl_expression = batch["ctrl_expression"].to(device)
            target_expression = batch["target_expression"].to(device)
            de_idx = batch["de_idx"].to(device)
            de_idx_len = batch["de_idx_len"].to(device)

            # Forward
            output = model(
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                pert_gene_indices=pert_gene_indices,
                ctrl_expression=ctrl_expression,
            )

            # Loss
            losses = compute_loss(
                pred=output["pred_expression"],
                target=target_expression,
                de_idx=de_idx,
                de_idx_len=de_idx_len,
                ctrl_expression=ctrl_expression,
                mse_weight=cfg.training.loss.mse_weight,
                de_mse_weight=cfg.training.loss.de_mse_weight,
                direction_weight=cfg.training.loss.direction_weight,
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            scheduler.step()

            # Accumulate
            epoch_loss += losses["total"].item()
            epoch_mse += losses["mse"].item()
            epoch_de_mse += losses["de_mse"].item()
            epoch_dir += losses["direction"].item()
            num_batches += 1

            if is_main_process(rank):
                pbar.set_postfix(
                    loss=f"{losses['total'].item():.4f}",
                    mse=f"{losses['mse'].item():.4f}",
                    lr=f"{scheduler.get_lr()[0]:.2e}",
                )

        # Epoch averages
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_mse = epoch_mse / max(num_batches, 1)
        avg_de_mse = epoch_de_mse / max(num_batches, 1)
        avg_dir = epoch_dir / max(num_batches, 1)

        if is_main_process(rank):
            logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f}, "
                f"mse={avg_mse:.4f}, de_mse={avg_de_mse:.4f}, dir={avg_dir:.4f}, "
                f"lr={scheduler.get_lr()[0]:.2e}"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/mse": avg_mse,
                    "train/de_mse": avg_de_mse,
                    "train/direction": avg_dir,
                    "train/lr": scheduler.get_lr()[0],
                    "epoch": epoch + 1,
                })

        # ------------------------------------------------------------------
        # Evaluation (rank 0 only)
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg.training.eval_every == 0:
            if is_main_process(rank):
                logger.info("Evaluating on validation set...")
                val_metrics = evaluate_model(
                    raw_model, val_dataset, device, cfg, collate_perturbation_batch
                )
                logger.info(
                    f"Val metrics: mse={val_metrics['mse']:.4f}, "
                    f"pearson_top20={val_metrics['pearson_top20']:.4f}, "
                    f"pearson_delta={val_metrics['pearson_delta_top20']:.4f}, "
                    f"dir_acc={val_metrics['direction_accuracy']:.4f}"
                )

                if use_wandb:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

                # Save best model
                if val_metrics["mse"] < best_val_mse:
                    best_val_mse = val_metrics["mse"]
                    patience_counter = 0
                    torch.save(
                        raw_model.state_dict(), output_dir / "best_model.pt"
                    )
                    logger.info(f"New best model saved (val_mse={best_val_mse:.4f})")
                else:
                    patience_counter += 1
                    logger.info(
                        f"No improvement ({patience_counter}/{cfg.training.early_stopping_patience})"
                    )

            # Broadcast early stopping decision from rank 0
            if distributed:
                stop_tensor = torch.tensor(
                    [1 if patience_counter >= cfg.training.early_stopping_patience else 0],
                    device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    if is_main_process(rank):
                        logger.info("Early stopping triggered.")
                    break
            else:
                if patience_counter >= cfg.training.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

            # Barrier so all ranks wait for eval/save to finish
            if distributed:
                dist.barrier()

    # ------------------------------------------------------------------
    # 8. Final test evaluation (rank 0 only)
    # ------------------------------------------------------------------
    if is_main_process(rank):
        logger.info("Loading best model for test evaluation...")
        raw_model.load_state_dict(
            torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
        )

        test_metrics = evaluate_model(
            raw_model, test_dataset, device, cfg, collate_perturbation_batch
        )
        logger.info("=" * 60)
        logger.info("TEST RESULTS:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

        # Print LangPert comparison table for Replogle datasets
        if dataset_name in ("replogle_k562_essential", "replogle_rpe1_essential"):
            comparison = format_comparison_table(test_metrics, dataset_name)
            logger.info("\n" + comparison)

        if use_wandb:
            import wandb
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()

        # Save test metrics
        import json
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    cleanup_ddp()


# =====================================================================
# Entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="DyGenePT Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
