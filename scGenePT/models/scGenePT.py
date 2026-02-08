# Model Class is based on, modifies and augments the TransformerGenerator model class 
# from https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/generation_model.py, 
# retrieved in July 2024.

import gc
import pickle as pkl
import matplotlib.pyplot as plt
from utils.scgpt_config import *

import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn, Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id
import os
import math
from typing import Mapping, Optional, Tuple, Any, Union

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from torch.utils.data import dataset
from tqdm import trange

from scgpt.model import (
    ExprDecoder,
    MVCDecoder,
    ContinuousValueEncoder,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
)

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")


class scGenePT(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        n_perturbagens: int,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        n_input_bins: Optional[int] = 0,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        decoder_activation: Optional[str] = None,
        decoder_adaptive_bias: bool = False,
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        embs_to_include = ['scGPT_counts_embs', 'scGPT_token_embs', 'genePT_token_embs'],
        genept_embs = None,
        genept_emb_type = None, 
        genept_emb_size = 1536,
        go_embs_to_include = None,
        go_emb_type = None,
        go_emb_size = 1536,
        proj_layer = None
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.embs_to_include = embs_to_include
        self.go_embs_to_include = go_embs_to_include
        self.go_emb_type = go_emb_type
        
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            try:
                from flash_attn.flash_attention import FlashMHA
            except ImportError:
                import warnings

                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer
        
        print(f'Using the following embeddings:{self.embs_to_include}')

        # scGPT gene token encoder
        if 'scGPT_token_embs' in self.embs_to_include:
            self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
            
        # scGPT counts encoder
        if 'scGPT_counts_embs' in self.embs_to_include:
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
            
        # genePT gene token encoder
        if 'genePT_token_embs_gpt' in self.embs_to_include:
            self.genept_encoder = GenePTEncoder(ntoken, d_model, padding_idx=vocab[pad_token], genept_lookup_embed=genept_embs, 
                                                genept_embs_size=genept_emb_size, proj_layer = proj_layer)
        # GO Annotations gene token encoder
        if 'GO_token_embs_gpt_concat' in self.embs_to_include or 'GO_token_embs_gpt_avg' in self.embs_to_include:
            if go_emb_type == 'c':
                self.gopt_encoder_c = GOPTEncoder(ntoken, d_model, padding_idx=vocab[pad_token], 
                                          gopt_lookup_embed=go_embs_to_include[go_emb_type], gopt_embs_size=go_emb_size)
            elif go_emb_type == 'f':
                self.gopt_encoder_f = GOPTEncoder(ntoken, d_model, padding_idx=vocab[pad_token], 
                                          gopt_lookup_embed=go_embs_to_include[go_emb_type], gopt_embs_size=go_emb_size)
            elif go_emb_type == 'p':
                self.gopt_encoder_p = GOPTEncoder(ntoken, d_model, padding_idx=vocab[pad_token], 
                                          gopt_lookup_embed=go_embs_to_include[go_emb_type], gopt_embs_size=go_emb_size)
            elif go_emb_type == 'all':
                self.gopt_encoder_f = GOPTEncoder(ntoken, d_model, padding_idx=vocab[pad_token], 
                                          gopt_lookup_embed=go_embs_to_include[go_emb_type], gopt_embs_size=go_emb_size)
        # Perturbation flags encoder 
        self.pert_encoder = nn.Embedding(n_perturbagens + 1, d_model, padding_idx=pert_pad_id)

        self.ln = nn.LayerNorm(d_model)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Gene Expression Decoder
        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
        )
                
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
            )
        if 'scGPT_token_embs' in self.embs_to_include:
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        """
        Encodes a sequence of src gene tokens and count values
        
        Args: 
            src: gene indices corresponding to the gene tokens to encode
            values: gene counts corresponding to the gene tokens in src
            input_pert_flags: perturbation flags corresponding to the genes in src; 1 if a gene is perturbed, 0 if not
            src_key_padding_mask: mask used during training for gene src tokens 
        """
                
        # Mapping from embedding types 2 values
        embs2values = {}
        
        # Encode the gene tokens using scGPT gene token encoder
        if 'scGPT_token_embs' in self.embs_to_include:
            src_scgpt = self.encoder(src)  # (batch, seq_len, embsize)
            embs2values['scGPT_token_embs'] = src_scgpt
        # Encode the counts using scGPT counts encoder
        if 'scGPT_counts_embs' in self.embs_to_include:
            embs2values['scGPT_counts_embs']  = self.value_encoder(values)  # (batch, seq_len, embsize)
        # Encode the gene tokens using the genePT encoder
        if 'genePT_token_embs_gpt' in self.embs_to_include or 'genePT_token_embs_llama' in self.embs_to_include:
            src_genept = self.genept_encoder(src)  # (batch, seq_len, embsize)
            embs2values['genePT_token_embs'] = src_genept
        # Encode the gene tokens using the GO embeddings encoder
        if 'GO_token_embs_gpt_avg' in self.embs_to_include or  'GO_token_embs_gpt_concat' in self.embs_to_include:
            if self.go_emb_type == 'c':
                src_go_embs = self.gopt_encoder_c(src)
            elif self.go_emb_type == 'p':
                src_go_embs = self.gopt_encoder_p(src)
            elif self.go_emb_type == 'f':
                src_go_embs = self.gopt_encoder_f(src)
            elif self.go_emb_type == 'all':
                src_go_embs = self.gopt_encoder_f(src)
            embs2values['GO_token_embs_' + self.go_emb_type] = src_go_embs
        
        # Encode the perturbation flags
        embs2values['pert_embs'] = self.pert_encoder(input_pert_flags)
             
        # Add all embeddings together
        seen_embs = False
        for emb, emb_value in embs2values.items():
            if not seen_embs:
                total_embs = emb_value
                seen_embs = True
            else:
                total_embs += emb_value
        total_embs = total_embs.type(torch.float32)
        total_embs = self.ln(total_embs)

        # Feed embeddings into transformer_encoder
        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    # Not modified from original scGPT architecture
    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass through the model 
        
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True
            logger.warning("Auto set do_sample to True when model is in eval mode.")

        # binning input gene values
        if self.n_input_bins > 0:
            from ..preprocess import binning

            processed_values = torch.stack(
                [binning(row, n_bins=self.n_input_bins) for row in values], dim
            ).to(values.device)
        else:
            processed_values = values

        transformer_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        mlm_output = self.decoder(transformer_output)
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        output_to_cpu: bool = True,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len]
            values: Tensor, shape [N, seq_len]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        outputs = []
        N = src.size(0)
        device = next(self.parameters()).device
        for i in trange(0, N, batch_size):
            output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
            )
            if output_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return torch.cat(outputs, dim=0)
    
    def pred_perturb_from_ctrl(
        self,
        adata_ctrl,
        perturbation,
        gene_names,
        device,
        gene_ids=None,
        amp=True,
        pool_size = None, 
        return_mean = True
    ) -> Tensor:
        """
        Perturbation prediction from a control sample
        
        Args:
            adata_ctrl: adata control sample to predict from
            perturbation: perturbation type, in str form; eg 'FOSB+ctrl', 'SAMD1+ZBTB1'
            gene_names: list of gene names in the dataset the model has been trained on
            include_zero_gene: True if to include zero genes
            gene_ids: gene_ids to predict for 
            pool_size: number of control samples to predict for; if None, predicts over all; otherwise, samples randomly for pool_size
            return_mean: if True, returns mean of prediction over control samples; else returns a list of all predictions

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        gene_ids = torch.tensor(gene_ids).long().unsqueeze(0).to(device)

        if pool_size == None:
            pool_size = len(adata_ctrl)

        ctrls = np.array(adata_ctrl[np.random.randint(0, len(adata_ctrl), pool_size)].X.toarray())
        src_key_padding_mask = torch.zeros_like(
            gene_ids, dtype=torch.bool, device=device
        )

        all_pred_gene_values = []
        for ori_gene_values in ctrls:
            pert_flags = np.zeros(len(ori_gene_values))

            if perturbation != 'ctrl':
                for x in perturbation.split('+'):
                    if x != 'ctrl':
                        pert_flags[gene_names.index(x)] = 1
            pert_flags = torch.from_numpy(pert_flags).long().to(device).unsqueeze(0)
            ori_gene_values = torch.from_numpy(np.expand_dims(ori_gene_values, 0)).to(dtype = torch.float32).to(device)
            # model = model.to(torch.float32)
            with torch.cuda.amp.autocast(enabled=amp):
                with torch.no_grad():
                    output_dict = self(
                        gene_ids,
                        ori_gene_values,
                        pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=True,
                    )
                pred_gene_values = output_dict["mlm_output"].float().detach().cpu().numpy()
                all_pred_gene_values.append(pred_gene_values)
        if return_mean:
            return np.mean(all_pred_gene_values, axis = 0)
        else:
            return np.array(all_pred_gene_values)
    
    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
        pert_type = 'intrinsic'
    ) -> Tensor:
        """
        Perturbation prediction for a given batch of data
        
        Args:
            batch_data: a dictionary of input data with keys
            include_zero_gene: True if to include zero genes
            gene_ids: gene_ids to predict for 
            pert_type: intrinsic or extrinsic, depending on perturbation type

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        if pert_type == 'intrinsic':
            batch_data.to(device)
            batch_size = len(batch_data.pert)
            x: torch.Tensor = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, -1)
        else:
            batch_size = len(batch_data['ctrl_gene_expression'])        
            ori_gene_values = batch_data['ctrl_gene_expression'].type(torch.float32) # control cell as input
            pert_flags = batch_data['pert_vector'].long() # flag whether a gene is perturbed or not

        ori_gene_values = ori_gene_values.to(device)
        pert_flags = pert_flags.to(device)

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                with torch.no_grad():
                    output_dict = self(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=True,
                    )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values

class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """
        Encodes a gene token during training using learned token representations. 
        Initialized with pre-trained scGPT gene token embeddings.
        
        Args: 
            num_embeddings: number of genes
            embedding_dim: dimension of the gene
            padding_idx: padding_idx for the Embedding
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x
                
class GenePTEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        proj_layer = None,
        padding_idx: Optional[int] = None,
        genept_lookup_embed: Optional = [],
        genept_embs_size = 1536
    ):
        """
        Encodes a gene token during training using textual genePT representations. 
        Initialized with pre-trained genePT gene annotations embedded using LLMs.
        
        Args: 
            num_embeddings: number of genes
            embedding_dim: dimension of the gene
            padding_idx: padding_idx for the Embedding
            genept_lookup_embed: pre-trained embeddings used to initialize the Embedding layer
            genept_embs_size: size of the pre-trained embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(genept_lookup_embed), freeze = False, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(genept_embs_size)
        if proj_layer:
            print("Using a learned projection layer")
            self.proj_layer = proj_layer
        else:
            self.proj_layer = nn.Linear(genept_embs_size, embedding_dim)
        self.enc_norm = nn.LayerNorm(embedding_dim)
       

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x).float()  # (batch, seq_len, embsize)
        x = self.proj_layer(x)
        x = self.enc_norm(x)
        return x
                
class GOPTEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        gopt_lookup_embed: Optional = [],
        gopt_embs_size = 1536
    ):
        """
        Encodes a gene token during training using textual GO annotations. 
        Initialized with pre-trained GO (Gene Ontology) gene annotations embedded using LLMs.
        
        Args: 
            num_embeddings: number of genes
            embedding_dim: dimension of the gene
            padding_idx: padding_idx for the Embedding
            genept_lookup_embed: pre-trained embeddings used to initialize the Embedding layer
            genept_embs_size: size of the pre-trained embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(gopt_lookup_embed), freeze = False, padding_idx=padding_idx)
        self.fc = nn.Linear(gopt_embs_size, embedding_dim)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x).float()  # (batch, seq_len, embsize)
        x = self.fc(x)
        x = self.enc_norm(x)
        return x

    
class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
    
    
def get_batch_data(batch_data, include_zero_gene, n_genes, max_seq_len, gene_ids, device):
    """
    Parses a batch of data from a PertData batch object.
    
    Args:
        include_zero_gene: true if to include zero genes
        n_genes: number of total genes in the sequence
        max_seq_len: max sequence length to use during training
        gene_ids: vocab indices of the n_genes in the sequence
        device: device used for training
        
    Returns:
        mapped_input_gene_ids: src token indices corresponding to sampled gene tokens
        input_values: gene count values corresponding to mapped_input_gene_ids
        input_pert_flags: perturbation flags corresponding to mapped_input_gene_ids; 1 if gene is perturbed, 0 if not
        src_key_padding_mask: mask for src token indices
        target_values: target post-perturbation values for sampled gene tokens corresponding to mapped_input_genes_ids
    """
    batch_size = len(batch_data.y)
    x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
    ori_gene_values = x[:, 0].view(batch_size, n_genes)
    pert_flags = x[:, 1].long().view(batch_size, n_genes)
    target_gene_values = batch_data.y  # (batch_size, n_genes)
        
    
    if include_zero_gene in ["all", "batch-wise"]:
        if include_zero_gene == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
        # sample input_gene_id
        if len(input_gene_ids) > max_seq_len:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                :max_seq_len
            ]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )
    return mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values

    
        
def train_epoch(model, train_loader, loss_fn, optimizer, 
                scheduler, logger, scaler, device, n_genes, gene_ids, 
                num_epoch, include_zero_gene, amp, dataset_name, 
                max_seq_len, log_interval, gene2idx = {}) -> None:
    """
    Trains the model for one epoch on train_loader.
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    
    for batch, batch_data in enumerate(train_loader):
        batch_data.to(device)
        mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values = get_batch_data(batch_data, include_zero_gene,
                                                                                                                    n_genes, max_seq_len, gene_ids, device)
        
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_fn(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            logger.info(
                f"| epoch {num_epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:7.5f}|"
            )
            total_loss = 0
            start_time = time.time()


def evaluate_on_epoch(model, val_loader, loss_fn, 
                      logger, scaler, device, n_genes, gene_ids, save_dir, 
                      include_zero_gene, amp, epoch, dataset_name, model_type, 
                      rnd_seed, loss_to_minimize, max_seq_len, log_interval, 
                      outputs_dir, gene2idx = {}) -> float:
    """
    Evaluates the model on MSE loss on validation loader
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_data.to(device)
            mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values = get_batch_data(batch_data, include_zero_gene, n_genes, max_seq_len, gene_ids, device)

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = loss_fn(output_values, target_values, masked_positions)
            total_loss += loss.item()
    mse_loss = total_loss / len(val_loader)
    metrics = {'val_mse' : mse_loss}    
    return metrics


def train_model(model, pert_data, epochs, loss_fn, 
                optimizer, scheduler, scaler, device,
                gene_ids, logger, include_zero_gene, amp, 
                dataset_name, model_type, rnd_seed, 
                max_seq_len, log_interval, early_stop, gene2idx = {}, 
                save_models_each_epoch = False, save_dir = "/tmp", loss_to_minimize = 'mse'):
    """
    Trains the model for a given number of epochs.
    """
   
    best_val_loss = float("inf")
    best_val_pearson_de = 0
    best_model = None
    patience = 0
    n_genes = len(gene_ids)

    for epoch in range(1, epochs + 1):
        outputs_dir = os.path.join(save_dir, "/metrics/val/val_metrics_detailed_epoch" + str(epoch) + ".json")
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        val_loader = pert_data.dataloader["val_loader"]

        # Train model on train_loader
        train_epoch(model, train_loader, loss_fn, optimizer, 
                    scheduler, logger, scaler, device, n_genes, 
                    gene_ids, epoch, include_zero_gene, amp, 
                    dataset_name, max_seq_len, log_interval, gene2idx)
        
        # Validate on val_loader
        val_metrics = evaluate_on_epoch(model, val_loader, loss_fn, logger, 
                                        scaler, device, n_genes, gene_ids, 
                                        save_dir, include_zero_gene, amp, epoch, 
                                        dataset_name, model_type, rnd_seed, 
                                        loss_to_minimize, max_seq_len, log_interval, 
                                        outputs_dir, gene2idx)
        
        elapsed = time.time() - epoch_start_time
        val_loss = val_metrics[f'val_{loss_to_minimize}']
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse_de {val_loss:7.4f} |"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {best_val_loss:5.7f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        if save_models_each_epoch:
            torch.save(
                model.state_dict(),
                save_dir / f"models/model_{epoch}.pt",
            )

        scheduler.step()
    return best_model