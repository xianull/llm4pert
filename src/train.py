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
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.utils import set_seed, get_device, setup_logging
from src.model.dygenept import DyGenePT
from src.data.dataset import PerturbationDataset
from src.data.collator import collate_perturbation_batch
from src.evaluate import evaluate_model, format_comparison_table, format_subgroup_table


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
    rank_weight: float = 0.0,
    autofocus_gamma: float = 0.0,
) -> dict:
    """Combined loss with GEARS-style autofocus and direction components.

    Autofocus loss (GEARS, Roohani et al. 2023):
        L_af = mean(|pred - target|^(2+gamma))
        When gamma > 0, automatically upweights DE genes (large errors)
        and downweights non-DE genes (small errors). No DE index needed.

    Direction loss (applied to ALL genes, not just top-20):
        L_dir = mean([sign(pred_delta) - sign(true_delta)]^2)
        Penalizes wrong direction of change relative to control.

    DE-focused MSE: additional MSE on known top DE genes (optional).
    """
    B, G = pred.shape
    device = pred.device

    # --- Autofocus loss (GEARS-style) ---
    if autofocus_gamma > 0:
        # |error|^(2+gamma): automatically focuses on large-error (DE) genes
        error = (pred - target).abs()
        loss_mse = (error ** (2 + autofocus_gamma)).mean()
    else:
        # Standard MSE
        loss_mse = nn.functional.mse_loss(pred, target)

    # --- Soft direction loss: tanh instead of sign (smooth, no dead zones) ---
    if direction_weight > 0:
        pred_delta = pred - ctrl_expression
        true_delta = target - ctrl_expression
        # Soft sign via tanh: continuous & differentiable everywhere
        k = 10.0  # temperature: higher = sharper (closer to sign), lower = smoother
        pred_soft = torch.tanh(k * pred_delta)
        true_soft = torch.tanh(k * true_delta)
        # Weighted by |true_delta|: DE genes dominate, noise genes ignored
        weights = true_delta.abs()
        loss_dir = ((pred_soft - true_soft) ** 2 * weights).sum() / weights.sum().clamp(min=1e-6)
    else:
        loss_dir = torch.tensor(0.0, device=device)

    # --- DE-focused MSE (optional, on top-20 DE genes) ---
    if de_mse_weight > 0 and de_idx_len.sum() > 0:
        de_idx_clamped = de_idx.clamp(0, G - 1)
        pred_de = pred.gather(1, de_idx_clamped)
        target_de = target.gather(1, de_idx_clamped)
        mask = torch.arange(de_idx.shape[1], device=device).unsqueeze(0)
        mask = (mask < de_idx_len.unsqueeze(1)).float()
        de_mse_raw = nn.functional.mse_loss(pred_de, target_de, reduction="none")
        loss_de_mse = (de_mse_raw * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss_de_mse = torch.tensor(0.0, device=device)

    # --- Ranking loss (optional) ---
    if rank_weight > 0 and de_idx_len.sum() > 0:
        de_idx_clamped = de_idx.clamp(0, G - 1)
        mask = torch.arange(de_idx.shape[1], device=device).unsqueeze(0)
        mask = (mask < de_idx_len.unsqueeze(1))
        loss_rank = _compute_rank_loss(pred, target, ctrl_expression, de_idx_clamped, mask)
    else:
        loss_rank = torch.tensor(0.0, device=device)

    total = (
        mse_weight * loss_mse
        + de_mse_weight * loss_de_mse
        + direction_weight * loss_dir
        + rank_weight * loss_rank
    )
    return {
        "total": total, "mse": loss_mse, "de_mse": loss_de_mse,
        "direction": loss_dir, "rank": loss_rank,
    }


def _compute_rank_loss(
    pred: torch.Tensor,            # (B, G)
    target: torch.Tensor,          # (B, G)
    ctrl_expression: torch.Tensor, # (B, G)
    de_idx: torch.LongTensor,     # (B, top_de)
    de_mask: torch.BoolTensor,    # (B, top_de)
    margin: float = 0.5,
    n_neg_samples: int = 50,
) -> torch.Tensor:
    """Margin ranking loss: |pred_delta| for DE genes > |pred_delta| for non-DE genes.

    For each DE gene, sample random non-DE genes and enforce that the
    predicted |delta| of the DE gene exceeds that of the non-DE gene by margin.
    """
    B, G = pred.shape
    device = pred.device

    pred_delta = (pred - ctrl_expression).abs()      # (B, G)
    target_delta = (target - ctrl_expression).abs()   # (B, G)

    # DE genes' predicted |delta|
    pred_de_delta = pred_delta.gather(1, de_idx)      # (B, top_de)

    # Sample random non-DE gene indices
    rand_idx = torch.randint(0, G, (B, n_neg_samples), device=device)  # (B, n_neg)
    pred_neg_delta = pred_delta.gather(1, rand_idx)    # (B, n_neg)

    # Pairwise margin ranking: each DE gene vs each neg sample
    # pos: (B, top_de, 1), neg: (B, 1, n_neg) -> (B, top_de, n_neg)
    pos = pred_de_delta.unsqueeze(-1)                  # (B, top_de, 1)
    neg = pred_neg_delta.unsqueeze(1)                   # (B, 1, n_neg)

    # hinge loss: max(0, margin - (pos - neg))
    rank_loss = torch.clamp(margin - (pos - neg), min=0)  # (B, top_de, n_neg)

    # Mask out padded DE positions
    rank_loss = rank_loss * de_mask.unsqueeze(-1).float()  # (B, top_de, n_neg)

    return rank_loss.sum() / (de_mask.float().sum() * n_neg_samples).clamp(min=1)


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
# Multi-dataset loading for joint training
# =====================================================================
def load_multi_dataset(cfg, rank, distributed):
    """Load multiple GEARS datasets and build unified gene vocabulary.

    Returns:
        union_gene_names: sorted list of all genes across datasets
        datasets_info: list of dicts, each with:
            - name: dataset name
            - pert_data: GEARS PertData
            - gene_names: local gene list
            - gene_remap: np.array mapping local → unified indices
            - pert_type_id: int
            - train_pert_genes: set of training perturbation gene names
            - subgroup: GEARS subgroup dict or None
    """
    import numpy as np
    from gears import PertData
    import gears.pertdata as _gears_pd

    # Monkey-patch GEARS
    _orig_get_pert_idx = _gears_pd.PertData.get_pert_idx
    def _safe_get_pert_idx(self, pert_category, adata_):
        try:
            return _orig_get_pert_idx(self, pert_category, adata_)
        except IndexError:
            pert_idx = []
            for p in pert_category:
                idx = np.where(p == self.gene_names)[0]
                pert_idx.append(idx[0] if len(idx) > 0 else -1)
            return pert_idx
    _gears_pd.PertData.get_pert_idx = _safe_get_pert_idx

    dataset_names = list(cfg.training.datasets)
    all_gene_sets = []
    datasets_info = []

    for ds_name in dataset_names:
        ds_cfg = cfg.dataset_configs[ds_name]
        data_name = ds_cfg.data_name
        split_mode = ds_cfg.get("split_mode", "simulation")
        pert_type_id = int(getattr(ds_cfg, 'pert_type', 0))

        if is_main_process(rank):
            print(f"[MultiDataset] Loading {data_name}...")

        pert_data = PertData(cfg.paths.perturb_data_dir)
        local_path = os.path.join(cfg.paths.perturb_data_dir, data_name)

        if is_main_process(rank):
            if os.path.exists(local_path):
                pert_data.load(data_path=local_path)
            else:
                pert_data.load(data_name=data_name)
            pert_data.prepare_split(split=split_mode, seed=cfg.training.seed)
        if distributed:
            dist.barrier()
        if not is_main_process(rank):
            if os.path.exists(local_path):
                pert_data.load(data_path=local_path)
            else:
                pert_data.load(data_name=data_name)
            pert_data.prepare_split(split=split_mode, seed=cfg.training.seed)

        # Gene names
        if "gene_name" in pert_data.adata.var.columns:
            gene_names = list(pert_data.adata.var["gene_name"])
        else:
            gene_names = list(pert_data.adata.var_names)

        all_gene_sets.append(set(gene_names))

        # Training perturbation genes
        train_pert_genes = set()
        if hasattr(pert_data, "set2conditions") and "train" in pert_data.set2conditions:
            for pname in pert_data.set2conditions["train"]:
                for g in pname.split("+"):
                    if g != "ctrl":
                        train_pert_genes.add(g)

        datasets_info.append({
            "name": ds_name,
            "pert_data": pert_data,
            "gene_names": gene_names,
            "pert_type_id": pert_type_id,
            "train_pert_genes": train_pert_genes,
            "subgroup": getattr(pert_data, "subgroup", None),
        })

    # Build union gene vocabulary
    union_genes = sorted(set.union(*all_gene_sets))
    union_g2i = {g: i for i, g in enumerate(union_genes)}

    if is_main_process(rank):
        print(f"[MultiDataset] Union: {len(union_genes)} genes from {len(dataset_names)} datasets")
        for info in datasets_info:
            local_n = len(info["gene_names"])
            overlap = len(set(info["gene_names"]) & set(union_genes))
            print(f"  {info['name']}: {local_n} genes ({overlap} in union)")

    # Build gene_remap for each dataset: local_idx → unified_idx
    for info in datasets_info:
        local_genes = info["gene_names"]
        remap = np.array([union_g2i[g] for g in local_genes], dtype=np.int64)
        info["gene_remap"] = remap

    return union_genes, datasets_info


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
    # 1-4. Load data, build model & datasets
    # ------------------------------------------------------------------
    import numpy as np

    multi_dataset = hasattr(cfg.training, 'datasets') and cfg.training.datasets

    if multi_dataset:
        # ============================================================
        # MULTI-DATASET MODE (joint K562 + RPE1)
        # ============================================================
        union_gene_names, datasets_info = load_multi_dataset(cfg, rank, distributed)
        num_genes = len(union_gene_names)
        gene_names = union_gene_names

        if is_main_process(rank):
            logger.info(f"Joint training: {num_genes} unified genes")

        # Load facet embeddings (must cover union genes)
        saved = torch.load(cfg.paths.facet_embeddings, map_location="cpu", weights_only=False)
        gene_to_facet_idx = saved["gene_to_idx"]
        if is_main_process(rank):
            logger.info(f"Loaded facet tensor with {len(gene_to_facet_idx)} genes")

        # Initialize model
        if is_main_process(rank):
            logger.info("Initializing DyGenePT model...")
        model = DyGenePT(cfg, num_genes=num_genes, gene_names=gene_names)
        model = model.to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank])
        raw_model = model.module if distributed else model

        # scGPT vocab for tokenization
        scgpt_vocab = getattr(raw_model.cell_encoder, 'vocab', None)
        if scgpt_vocab is None:
            from scgpt.tokenizer import GeneVocab
            scgpt_vocab = GeneVocab.from_file(Path(cfg.paths.scgpt_checkpoint) / "vocab.json")

        pace_cfg = getattr(cfg, 'pert_aware_cell_encoding', None)
        pert_aware_enabled = pace_cfg is not None and getattr(pace_cfg, 'enabled', False)
        force_include_pert = getattr(pace_cfg, 'force_include_pert_genes', True) if pace_cfg else True
        pert_gene_dropout = float(getattr(cfg.training, 'pert_gene_dropout', 0.0))

        # Build per-dataset train/val/test with gene remapping
        if is_main_process(rank):
            print("Creating dataloaders (multi-dataset)...")

        train_datasets = []
        val_datasets = {}    # {ds_name: dataset}
        test_datasets = {}   # {ds_name: dataset}
        all_train_pert_genes = set()
        pert_subgroup = None  # not used for single-gene Replogle

        for info in datasets_info:
            ds_common = dict(
                gene_to_facet_idx=gene_to_facet_idx,
                scgpt_vocab=scgpt_vocab,
                gene_names=info["gene_names"],
                max_seq_len=cfg.cell_encoder.max_seq_len,
                pert_type_id=info["pert_type_id"],
                pert_aware=pert_aware_enabled,
                force_include_pert_genes=force_include_pert,
                gene_remap=info["gene_remap"],
                num_genes_out=num_genes,
            )

            train_ds = PerturbationDataset(
                info["pert_data"], "train", **ds_common,
                pert_gene_dropout=pert_gene_dropout,
            )
            val_ds = PerturbationDataset(info["pert_data"], "val", **ds_common)
            test_ds = PerturbationDataset(info["pert_data"], "test", **ds_common)

            train_datasets.append(train_ds)
            val_datasets[info["name"]] = val_ds
            test_datasets[info["name"]] = test_ds
            all_train_pert_genes.update(info["train_pert_genes"])

        train_dataset = ConcatDataset(train_datasets)
        train_pert_genes = all_train_pert_genes

        # For early stopping, use the first dataset's val set
        # (will evaluate all datasets during val)
        val_dataset = None  # signal to use multi-dataset eval

    else:
        # ============================================================
        # SINGLE-DATASET MODE (legacy)
        # ============================================================
        from gears import PertData
        import gears.pertdata as _gears_pd

        dataset_name = cfg.training.dataset
        if hasattr(cfg, "dataset_configs") and dataset_name in cfg.dataset_configs:
            ds_cfg = cfg.dataset_configs[dataset_name]
            data_name = ds_cfg.data_name
            split_mode = ds_cfg.get("split_mode", "simulation")
            pert_type_id = int(getattr(ds_cfg, 'pert_type', 0))
        else:
            data_name = dataset_name
            split_mode = "simulation"
            pert_type_id = 0

        if is_main_process(rank):
            logger.info(f"Loading GEARS dataset: {data_name} (split_mode={split_mode})")

        # Monkey-patch GEARS
        _orig_get_pert_idx = _gears_pd.PertData.get_pert_idx
        def _safe_get_pert_idx(self, pert_category, adata_):
            try:
                return _orig_get_pert_idx(self, pert_category, adata_)
            except IndexError:
                pert_idx = []
                for p in pert_category:
                    idx = np.where(p == self.gene_names)[0]
                    pert_idx.append(idx[0] if len(idx) > 0 else -1)
                return pert_idx
        _gears_pd.PertData.get_pert_idx = _safe_get_pert_idx

        pert_data = PertData(cfg.paths.perturb_data_dir)
        local_path = os.path.join(cfg.paths.perturb_data_dir, data_name)
        if is_main_process(rank):
            if os.path.exists(local_path):
                pert_data.load(data_path=local_path)
            else:
                pert_data.load(data_name=data_name)
            pert_data.prepare_split(split=split_mode, seed=cfg.training.seed)
        if distributed:
            dist.barrier()
        if not is_main_process(rank):
            if os.path.exists(local_path):
                pert_data.load(data_path=local_path)
            else:
                pert_data.load(data_name=data_name)
            pert_data.prepare_split(split=split_mode, seed=cfg.training.seed)

        pert_subgroup = getattr(pert_data, "subgroup", None)
        train_pert_genes = set()
        if hasattr(pert_data, "set2conditions") and "train" in pert_data.set2conditions:
            for pname in pert_data.set2conditions["train"]:
                for g in pname.split("+"):
                    if g != "ctrl":
                        train_pert_genes.add(g)

        if "gene_name" in pert_data.adata.var.columns:
            gene_names = list(pert_data.adata.var["gene_name"])
        else:
            gene_names = list(pert_data.adata.var_names)
        num_genes = len(gene_names)
        if is_main_process(rank):
            logger.info(f"Number of genes: {num_genes}")

        saved = torch.load(cfg.paths.facet_embeddings, map_location="cpu", weights_only=False)
        gene_to_facet_idx = saved["gene_to_idx"]
        if is_main_process(rank):
            logger.info(f"Loaded facet tensor with {len(gene_to_facet_idx)} genes")

        if is_main_process(rank):
            logger.info("Initializing DyGenePT model...")
        model = DyGenePT(cfg, num_genes=num_genes, gene_names=gene_names)
        model = model.to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank])
        raw_model = model.module if distributed else model

        scgpt_vocab = getattr(raw_model.cell_encoder, 'vocab', None)
        if scgpt_vocab is None:
            from scgpt.tokenizer import GeneVocab
            scgpt_vocab = GeneVocab.from_file(Path(cfg.paths.scgpt_checkpoint) / "vocab.json")

        pace_cfg = getattr(cfg, 'pert_aware_cell_encoding', None)
        pert_aware_enabled = pace_cfg is not None and getattr(pace_cfg, 'enabled', False)
        force_include_pert = getattr(pace_cfg, 'force_include_pert_genes', True) if pace_cfg else True
        pert_gene_dropout = float(getattr(cfg.training, 'pert_gene_dropout', 0.0))

        if is_main_process(rank):
            print("Creating dataloaders....")
        train_dataset = PerturbationDataset(
            pert_data, "train", gene_to_facet_idx, scgpt_vocab, gene_names,
            max_seq_len=cfg.cell_encoder.max_seq_len,
            pert_type_id=pert_type_id,
            pert_aware=pert_aware_enabled,
            force_include_pert_genes=force_include_pert,
            pert_gene_dropout=pert_gene_dropout,
        )
        val_dataset = PerturbationDataset(
            pert_data, "val", gene_to_facet_idx, scgpt_vocab, gene_names,
            max_seq_len=cfg.cell_encoder.max_seq_len,
            pert_type_id=pert_type_id,
            pert_aware=pert_aware_enabled,
            force_include_pert_genes=force_include_pert,
        )
        test_dataset = PerturbationDataset(
            pert_data, "test", gene_to_facet_idx, scgpt_vocab, gene_names,
            max_seq_len=cfg.cell_encoder.max_seq_len,
            pert_type_id=pert_type_id,
            pert_aware=pert_aware_enabled,
            force_include_pert_genes=force_include_pert,
        )
        val_datasets = None
        test_datasets = None

    # DataLoader (shared for both modes)
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
        print(f"Done! Train samples: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # 5. Optimizer and scheduler
    # ------------------------------------------------------------------
    # Separate parameter groups: gene_embed_proj gets lower lr to preserve
    # pre-trained semantic structure from BioLORD embeddings
    gene_embed_proj_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gene_embed_proj" in name:
            gene_embed_proj_params.append(param)
        else:
            other_params.append(param)

    gene_embed_lr_scale = float(getattr(cfg.decoder, 'gene_embed_lr_scale', 0.1))
    param_groups = [
        {"params": other_params, "lr": cfg.training.lr},
    ]
    if gene_embed_proj_params:
        param_groups.append({
            "params": gene_embed_proj_params,
            "lr": cfg.training.lr * gene_embed_lr_scale,
        })
        if is_main_process(rank):
            logger.info(
                f"gene_embed_proj lr = {cfg.training.lr * gene_embed_lr_scale:.2e} "
                f"(scale={gene_embed_lr_scale})"
            )

    optimizer = AdamW(
        param_groups,
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
    best_val_score = float("-inf")
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
            pert_type_ids = batch["pert_type_ids"].to(device)
            pert_flags = batch["pert_flags"].to(device) if "pert_flags" in batch else None

            # Forward
            output = model(
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                pert_gene_indices=pert_gene_indices,
                ctrl_expression=ctrl_expression,
                pert_type_ids=pert_type_ids,
                pert_flags=pert_flags,
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
                rank_weight=float(getattr(cfg.training.loss, 'rank_weight', 0.0)),
                autofocus_gamma=float(getattr(cfg.training.loss, 'autofocus_gamma', 0.0)),
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

                # Multi-dataset eval: evaluate each dataset separately, average for early stopping
                if val_datasets is not None:
                    all_scores = []
                    val_metrics = {}
                    for ds_name, ds in val_datasets.items():
                        ds_metrics = evaluate_model(
                            raw_model, ds, device, cfg, collate_perturbation_batch,
                            subgroup=None, train_pert_genes=train_pert_genes,
                        )
                        score = ds_metrics["pearson_delta_all"]
                        all_scores.append(score)
                        logger.info(
                            f"  [{ds_name}] P_d_all={score:.4f}, "
                            f"P_d_top20={ds_metrics['pearson_delta_top20']:.4f}, "
                            f"dir_acc={ds_metrics['direction_accuracy']:.4f}"
                        )
                        for k, v in ds_metrics.items():
                            val_metrics[f"{ds_name}/{k}"] = v
                        # Add dataset as a top-level "subgroup" for table display
                        for mk in ["pearson_delta_all", "pearson_delta_top20", "pearson_top20",
                                    "mse", "mse_de", "mae", "direction_accuracy"]:
                            if mk in ds_metrics:
                                val_metrics[f"{ds_name}/{mk}"] = ds_metrics[mk]
                        # Count: total perturbations evaluated in this dataset
                        n_total = sum(int(v) for k, v in ds_metrics.items()
                                      if k.startswith("_n_") and isinstance(v, (int, float)))
                        val_metrics[f"_n_{ds_name}"] = n_total
                        if use_wandb:
                            import wandb
                            wandb.log({f"val/{ds_name}/{k}": v for k, v in ds_metrics.items()})
                    # Average ALL metrics across datasets (not just a few)
                    ds_names_list = list(val_datasets.keys())
                    all_metric_keys = set()
                    for dn in ds_names_list:
                        for k in val_metrics:
                            if k.startswith(f"{dn}/") and not k.startswith(f"{dn}/_"):
                                all_metric_keys.add(k.split("/", 1)[1])
                    for mk in all_metric_keys:
                        vals = [val_metrics.get(f"{dn}/{mk}", 0) for dn in ds_names_list
                                if f"{dn}/{mk}" in val_metrics]
                        val_metrics[mk] = sum(vals) / len(vals) if vals else 0
                    # Also store per-dataset N counts for table
                    for dn in ds_names_list:
                        n_key = f"_n_{dn}"
                        # Sum all _n_ keys from this dataset
                        total_n = sum(v for k, v in val_metrics.items()
                                      if k.startswith(f"{dn}/_n_") and isinstance(v, (int, float)))
                        val_metrics[n_key] = int(total_n) if total_n else 0
                else:
                    val_metrics = evaluate_model(
                        raw_model, val_dataset, device, cfg, collate_perturbation_batch,
                        subgroup=pert_subgroup, train_pert_genes=train_pert_genes,
                    )
                logger.info(
                    f"Val metrics: mse={val_metrics['mse']:.4f}, "
                    f"pearson_top20={val_metrics['pearson_top20']:.4f}, "
                    f"pearson_delta={val_metrics['pearson_delta_top20']:.4f}, "
                    f"pearson_delta_all={val_metrics['pearson_delta_all']:.4f}, "
                    f"dir_acc={val_metrics['direction_accuracy']:.4f}"
                )
                # Per-subgroup breakdown
                subgroup_table = format_subgroup_table(val_metrics)
                if subgroup_table:
                    logger.info("\n" + subgroup_table)

                if use_wandb:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

                # Save best model (select by pearson_delta_all, higher is better)
                val_score = val_metrics["pearson_delta_all"]
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    ckpt_path = output_dir / "best_model.pt"
                    torch.save(
                        raw_model.state_dict(), ckpt_path
                    )
                    logger.info(f"New best model saved (pearson_delta_all={best_val_score:.4f})")
                    if use_wandb:
                        import wandb
                        wandb.save(str(ckpt_path), base_path=str(output_dir))
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

        test_metrics = {}
        if test_datasets is not None:
            # Multi-dataset test evaluation
            for ds_name, ds in test_datasets.items():
                ds_metrics = evaluate_model(
                    raw_model, ds, device, cfg, collate_perturbation_batch,
                    subgroup=None, train_pert_genes=train_pert_genes,
                )
                logger.info(f"\n{'='*60}")
                logger.info(f"TEST RESULTS — {ds_name}:")
                for k, v in ds_metrics.items():
                    if not k.startswith("_"):
                        logger.info(f"  {k}: {v:.4f}")
                        test_metrics[f"{ds_name}/{k}"] = v
                logger.info("=" * 60)

                # N count for table
                n_total = sum(int(v) for k, v in ds_metrics.items()
                              if k.startswith("_n_") and isinstance(v, (int, float)))
                test_metrics[f"_n_{ds_name}"] = n_total

                subgroup_table = format_subgroup_table(ds_metrics)
                if subgroup_table:
                    logger.info("\n" + subgroup_table)

            # Average all metrics across datasets
            ds_names_list = list(test_datasets.keys())
            all_metric_keys = set()
            for dn in ds_names_list:
                for k in test_metrics:
                    if k.startswith(f"{dn}/") and not k.startswith(f"{dn}/_"):
                        all_metric_keys.add(k.split("/", 1)[1])
            for mk in all_metric_keys:
                vals = [test_metrics[f"{dn}/{mk}"] for dn in ds_names_list
                        if f"{dn}/{mk}" in test_metrics]
                test_metrics[mk] = sum(vals) / len(vals) if vals else 0
        else:
            test_metrics = evaluate_model(
                raw_model, test_dataset, device, cfg, collate_perturbation_batch,
                subgroup=pert_subgroup, train_pert_genes=train_pert_genes,
            )
            logger.info("=" * 60)
            logger.info("TEST RESULTS:")
            for k, v in test_metrics.items():
                if not k.startswith("_"):
                    logger.info(f"  {k}: {v:.4f}")
            logger.info("=" * 60)

        # Print per-subgroup breakdown table
        if not multi_dataset:
            subgroup_table = format_subgroup_table(test_metrics)
            if subgroup_table:
                logger.info("\n" + subgroup_table)

        # Print LangPert comparison table for Replogle datasets
        if multi_dataset:
            for ds_name in test_datasets:
                if ds_name in ("replogle_k562_essential", "replogle_rpe1_essential"):
                    # Extract per-dataset metrics
                    ds_test = {k.split("/", 1)[1]: v
                               for k, v in test_metrics.items()
                               if k.startswith(f"{ds_name}/")}
                    if ds_test:
                        comparison = format_comparison_table(ds_test, ds_name)
                        logger.info("\n" + comparison)
        elif dataset_name in ("replogle_k562_essential", "replogle_rpe1_essential"):
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
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dot notation, e.g. cell_encoder.model_name=mlp"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply command-line overrides (e.g. cell_encoder.model_name=mlp)
    if args.overrides:
        from omegaconf import OmegaConf
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.resolve(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
