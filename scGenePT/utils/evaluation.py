from torch import nn, Tensor
from torch_geometric.loader import DataLoader
from typing import Iterable, List, Tuple, Dict, Union, Optional
import torch
import numpy as np
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
import json
import matplotlib.pyplot as plt
from gears.utils import create_cell_graph_dataset_for_prediction

def compute_test_metrics(pert_data, best_model, loader_type, save_dir, device, include_zero_gene, gene_ids, epoch = 'best'):
    test_loader = pert_data.dataloader[loader_type + "_loader"]
    print("Loaded dataloader!")
    test_res = eval_perturb(test_loader, best_model, device, include_zero_gene, gene_ids)
    print('Finished eval perturb')
    test_metrics, test_pert_res = compute_metrics(test_res)
    new_test_metrics = {}
    for k, v in test_metrics.items():
        print(loader_type + "_" + k + ": " + str(v))
        new_test_metrics[loader_type + "_" + k] = v

    # save the dicts in json
    with open(f"{save_dir}/metrics/{loader_type}/metrics_epoch_{epoch}.json", "w") as f:
        json.dump(test_metrics, f)
    with open(f"{save_dir}/metrics/{loader_type}/pert_res_epoch_{epoch}.json", "w") as f:
        json.dump(test_pert_res, f)

    deeper_res = deeper_analysis(pert_data.adata, test_res)
    non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

    metrics = ["pearson_delta", "pearson_delta_de"]
    metrics_non_dropout = [
        "pearson_delta_top20_de_non_dropout",
        "pearson_top20_de_non_dropout",
    ]
    subgroup_analysis = {}
    for name in pert_data.subgroup[loader_type + "_subgroup"].keys():
        subgroup_analysis[name] = {}
        for m in metrics:
            subgroup_analysis[name][m] = []

        for m in metrics_non_dropout:
            subgroup_analysis[name][m] = []

    for name, pert_list in pert_data.subgroup[loader_type + "_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(deeper_res[pert][m])

            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(non_dropout_res[pert][m])

    for name, result in subgroup_analysis.items():
        for m in result.keys():
            subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
            m_type = loader_type + "_" + name + "_" + m
            m_value = subgroup_analysis[name][m]
            if m_value == m_value:
                print(m_type + ": " + str(m_value))
                new_test_metrics[m_type] = str(m_value)
    return new_test_metrics


def eval_perturb(
    loader: DataLoader, model, device, include_zero_gene, gene_ids
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    print(len(loader))
    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results