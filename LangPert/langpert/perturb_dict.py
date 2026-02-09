"""Local PerturbDict implementation for loading perturbation data from h5ad files."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class PerturbDict:
    """Load and manage perturbation data from h5ad files.

    Replaces the external perturbdict dependency. Reads perturb_processed.h5ad
    and associated split pickle files.
    """

    def __init__(self):
        self._gene_names: Optional[List[str]] = None
        self._gene_ids: Optional[List[str]] = None
        self._gene_id_to_idx: Optional[Dict[str, int]] = None
        self._expression: Optional[csr_matrix] = None
        self._condition_codes: Optional[np.ndarray] = None
        self._condition_names: Optional[List[str]] = None
        self._control_mask: Optional[np.ndarray] = None
        self._ctrl_mean: Optional[np.ndarray] = None
        self._de_top20: Optional[Dict[str, List[str]]] = None
        self._rank_genes: Optional[Dict[str, List[str]]] = None
        self._uns_key_map: Optional[Dict[str, str]] = None
        self._data_dir: Optional[Path] = None
        self._dataset_name: Optional[str] = None
        self._cond_to_cell_indices: Optional[Dict[str, np.ndarray]] = None

    def load(self, path: str) -> "PerturbDict":
        """Load data from a directory containing perturb_processed.h5ad.

        Args:
            path: Path to the dataset directory (containing perturb_processed.h5ad)
                  or path to the h5ad file directly.
        """
        path = Path(path)
        if path.is_file() and path.suffix == ".h5ad":
            h5ad_path = path
            self._data_dir = path.parent
        elif path.is_dir():
            h5ad_path = path / "perturb_processed.h5ad"
            self._data_dir = path
        elif path.suffix == ".pkl":
            # Legacy path format: /path/to/dataset/pert_dict.pkl
            # Treat parent dir as the data dir
            self._data_dir = path.parent
            h5ad_path = self._data_dir / "perturb_processed.h5ad"
        else:
            raise FileNotFoundError(f"Cannot find data at {path}")

        if not h5ad_path.exists():
            raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")

        self._dataset_name = self._data_dir.name
        logger.info("Loading h5ad from %s", h5ad_path)

        with h5py.File(h5ad_path, "r") as f:
            self._load_expression(f)
            self._load_gene_info(f)
            self._load_conditions(f)
            self._load_control(f)
            self._load_de_info(f)

        self._build_condition_index()
        logger.info(
            "Loaded %d cells x %d genes, %d conditions",
            self._expression.shape[0],
            self._expression.shape[1],
            len(self._condition_names),
        )
        return self

    # ── Data loading helpers ────────────────────────────────────────

    def _load_expression(self, f: h5py.File):
        """Load sparse expression matrix."""
        data = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr = f["X/indptr"][:]
        n_cells = len(indptr) - 1
        # Infer n_genes from max index
        n_genes = int(indices.max()) + 1 if len(indices) > 0 else 0
        self._expression = csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))

    def _load_gene_info(self, f: h5py.File):
        """Load gene names and IDs, handling both data formats."""
        n_genes = self._expression.shape[1]

        # Gene names — two possible formats
        if "var/gene_name/categories" in f:
            # Format 1: replogle style — var/gene_name/{categories, codes}
            cats = [c.decode() if isinstance(c, bytes) else c for c in f["var/gene_name/categories"][:]]
            codes = f["var/gene_name/codes"][:]
            self._gene_names = [cats[c] for c in codes]
        elif "var/__categories/gene_name" in f:
            # Format 2: norman style — var/__categories/gene_name + var/gene_name (int codes)
            cats = [c.decode() if isinstance(c, bytes) else c for c in f["var/__categories/gene_name"][:]]
            codes = f["var/gene_name"][:]
            self._gene_names = [cats[c] for c in codes]
        else:
            # Fallback: direct dataset
            self._gene_names = [
                g.decode() if isinstance(g, bytes) else str(g) for g in f["var/gene_name"][:]
            ]

        # Gene IDs (Ensembl)
        gene_id_ds = f["var/gene_id"]
        if isinstance(gene_id_ds, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else c for c in gene_id_ds["categories"][:]]
            codes = gene_id_ds["codes"][:]
            self._gene_ids = [cats[c] for c in codes]
        else:
            self._gene_ids = [g.decode() if isinstance(g, bytes) else str(g) for g in gene_id_ds[:]]

        self._gene_id_to_idx = {gid: i for i, gid in enumerate(self._gene_ids)}

    def _load_conditions(self, f: h5py.File):
        """Load condition annotations."""
        if "obs/condition/categories" in f:
            # replogle format
            cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs/condition/categories"][:]]
            codes = f["obs/condition/codes"][:]
        elif "obs/__categories/condition" in f:
            # norman format
            cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs/__categories/condition"][:]]
            codes = f["obs/condition"][:]
        else:
            raise ValueError("Cannot find condition annotations in h5ad")

        self._condition_names = cats
        self._condition_codes = codes

    def _load_control(self, f: h5py.File):
        """Load control cell mask."""
        if "obs/control/categories" in f:
            # replogle format — categorical
            ctrl_cats = f["obs/control/categories"][:]
            ctrl_codes = f["obs/control/codes"][:]
            # control == 1 means it IS a control cell
            ctrl_val_idx = np.where(ctrl_cats == 1)[0]
            if len(ctrl_val_idx) > 0:
                self._control_mask = ctrl_codes == ctrl_val_idx[0]
            else:
                self._control_mask = np.zeros(len(ctrl_codes), dtype=bool)
        else:
            # norman format — direct int64
            ctrl = f["obs/control"][:]
            self._control_mask = ctrl == 1

    def _load_de_info(self, f: h5py.File):
        """Load differentially expressed gene rankings."""
        # Build condition name → uns key mapping
        self._uns_key_map = {}
        self._de_top20 = {}
        self._rank_genes = {}

        if "uns/top_non_dropout_de_20" in f:
            de_group = f["uns/top_non_dropout_de_20"]
            for uns_key in de_group.keys():
                # uns key format: {CellType}_{condition}_1+1
                cond_name = self._uns_key_to_condition(uns_key)
                self._uns_key_map[cond_name] = uns_key
                vals = de_group[uns_key][:]
                self._de_top20[cond_name] = [v.decode() if isinstance(v, bytes) else str(v) for v in vals]

        if "uns/rank_genes_groups_cov_all" in f:
            rgg_group = f["uns/rank_genes_groups_cov_all"]
            for uns_key in rgg_group.keys():
                cond_name = self._uns_key_to_condition(uns_key)
                vals = rgg_group[uns_key][:]
                self._rank_genes[cond_name] = [v.decode() if isinstance(v, bytes) else str(v) for v in vals]

    @staticmethod
    def _uns_key_to_condition(uns_key: str) -> str:
        """Convert uns key like 'K562_AAMP+ctrl_1+1' to condition name 'AAMP+ctrl'."""
        # Remove trailing '_1+1' or similar suffix
        # Format: {CellType}_{Condition}_{dose}
        # Split from the right: last part is dose info
        parts = uns_key.rsplit("_", 2)
        if len(parts) >= 3:
            # parts[0] = CellType, parts[1] = Condition, parts[2] = dose
            # But condition itself may contain underscores...
            # More robust: strip CellType prefix and dose suffix
            # Find first underscore (after cell type)
            first_underscore = uns_key.index("_")
            # Find the _N+N suffix pattern
            remainder = uns_key[first_underscore + 1 :]
            # The suffix is always _<digit>+<digit>
            # Find last occurrence of _<digit>+
            import re

            m = re.match(r"(.+)_(\d+\+\d+)$", remainder)
            if m:
                return m.group(1)
        return uns_key

    def _build_condition_index(self):
        """Pre-compute cell indices for each condition."""
        self._cond_to_cell_indices = {}
        for cond_idx, cond_name in enumerate(self._condition_names):
            cell_indices = np.where(self._condition_codes == cond_idx)[0]
            if len(cell_indices) > 0:
                self._cond_to_cell_indices[cond_name] = cell_indices

    # ── Public API ──────────────────────────────────────────────────

    @property
    def gene_names(self) -> List[str]:
        """List of gene names."""
        return self._gene_names

    def get_ctrl_mean(self) -> np.ndarray:
        """Mean expression of control cells."""
        if self._ctrl_mean is None:
            ctrl_cells = self._expression[self._control_mask]
            self._ctrl_mean = np.asarray(ctrl_cells.mean(axis=0)).flatten()
        return self._ctrl_mean

    def get_split_data(
        self, k: int = 5, fold: int = 0, seed: int = 42
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get train/test split using sklearn KFold — matches PerturbDict exactly.

        Algorithm (from PerturbDict/perturbdict/splits.py):
          1. sorted_perts = sorted(list(self.perturbations))
          2. KFold(n_splits=k, shuffle=True, random_state=seed)
          3. Return train_perts[fold], test_perts[fold]
        """
        from sklearn.model_selection import KFold

        all_conditions = sorted([
            name for name in self._cond_to_cell_indices
            if name != "ctrl" and not name.startswith("ctrl")
        ])
        perts = np.array(all_conditions)

        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        all_folds = list(kf.split(perts))
        train_idx, test_idx = all_folds[fold]

        train_perts = set(perts[train_idx])
        test_perts = set(perts[test_idx])

        train_data = self._compute_condition_means(train_perts)
        test_data = self._compute_condition_means(test_perts)

        logger.info(
            "KFold CV: k=%d, fold=%d, seed=%d -> %d train, %d test perturbations",
            k, fold, seed, len(train_data), len(test_data),
        )
        return train_data, test_data

    def get_de_mask(self, pert_name: str, k: int = 20) -> np.ndarray:
        """Boolean mask for top-k differentially expressed genes.

        Args:
            pert_name: Perturbation/condition name.
            k: Number of top DE genes.

        Returns:
            Boolean array of shape (n_genes,).
        """
        n_genes = len(self._gene_names)
        mask = np.zeros(n_genes, dtype=bool)

        if k == 20 and pert_name in self._de_top20:
            gene_ids = self._de_top20[pert_name]
        elif pert_name in self._rank_genes:
            gene_ids = self._rank_genes[pert_name][:k]
        else:
            logger.warning("No DE info for %s, returning empty mask", pert_name)
            return mask

        for gid in gene_ids:
            idx = self._gene_id_to_idx.get(gid)
            if idx is not None:
                mask[idx] = True

        return mask

    # ── Internal helpers ────────────────────────────────────────────

    def _find_split_file(self, seed: int) -> Path:
        """Locate the split pickle file."""
        splits_dir = self._data_dir / "splits"
        if not splits_dir.exists():
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

        # Pattern: {dataset_name}_simulation_{seed}_0.75.pkl
        expected = splits_dir / f"{self._dataset_name}_simulation_{seed}_0.75.pkl"
        if expected.exists():
            return expected

        # Fallback: find any matching file
        candidates = list(splits_dir.glob(f"*_simulation_{seed}_*.pkl"))
        # Prefer non-subgroup files
        non_subgroup = [c for c in candidates if "subgroup" not in c.name]
        if non_subgroup:
            return non_subgroup[0]
        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            f"No split file found for seed={seed} in {splits_dir}. "
            f"Available: {[f.name for f in splits_dir.glob('*.pkl')]}"
        )

    def _compute_condition_means(self, condition_names: set) -> Dict[str, np.ndarray]:
        """Compute mean expression for each condition."""
        result = {}
        for cond_name in condition_names:
            if cond_name not in self._cond_to_cell_indices:
                logger.warning("Condition '%s' not found in data, skipping", cond_name)
                continue
            cell_idx = self._cond_to_cell_indices[cond_name]
            mean_expr = np.asarray(self._expression[cell_idx].mean(axis=0)).flatten()
            result[cond_name] = mean_expr
        return result
