from utils.data_loading import *
from utils.scgpt_config import *

from models.scGenePT import *
import argparse
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pickle as pkl

def set_seed(seed):
    """
    Sets random seed
    
    Args:
        seed: random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_output_dirs(save_dir):
    """
    Creates sub-directories under save_dir where model outputs and evaluation metrics will be saved to.
    
    Args:
        save_dir: parent directory under which sub-directories are saved
    """
    metrics_dir_val = save_dir / "metrics/val"
    metrics_dir_test = save_dir / "metrics/test"
    models_dir = save_dir / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir_val.mkdir(parents=True, exist_ok=True)
    metrics_dir_test.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")

    
def load_dataloader(dataset_name, batch_size, val_batch_size, split = 'simulation'):
    """
    Loads data in a PertData format. Uses GEARS dataloaders implementation as described under https://github.com/snap-stanford/GEARS
    
    Args:
        dataset_name: name of the dataset to load. Example: 'adamson', 'norman'.
        batch_size: batch_size for the train datalaoder
        val_batch_size: batch_size for the validation dataloader
        split: split to use; tested with 'simulation'
        
    Returns:
        pert_data: PertData object
    """
    pert_data = PertData("data/")
    pert_data.load(data_name=dataset_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=val_batch_size)
    return pert_data

    
def get_args():
    """
    Parses command line arguments
    
    Returns:
        list of args
    """
    parser = argparse.ArgumentParser(description='Arguments for training ...')
    parser.add_argument(
        '--num-epochs', 
        type=int, 
        help='number of epochs to train the model for', 
        default = 20
    )
    parser.add_argument(
        '--model-type', 
        type=str, 
        help='Type of model to train. One of: [scgpt, scgenept_ncbi_gpt, scgenept_ncbi+uniprot_gpt, scgenept_go_c_gpt_concat, scgenept_go_f_gpt_concat, scgenept_go_p_gpt_concat, scgenept_go_all_gpt_concat, genept_ncbi_gpt, genept_ncbi+uniprot_gpt, go_c_gpt_concat, go_f_gpt_concat, go_p_gpt_concat, go_all_gpt_concat ...]. For full list of possible models, please visit https://github.com/czi-ai/scGenePT.', 
        default = "scgenept_ncbi_gpt"
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        help='train batch_size',
        default = 64
    )
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        help='val batch_size', 
        default = 64
    )
    parser.add_argument(
        '--device', 
        type=str, 
        help='device', 
        default = 'cuda:0'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='dataset to train on; ', 
        default = 'norman'
    )
    parser.add_argument(
        '--rnd-seed', 
        type=int, 
        help='random seed', 
        default = 42 # we run models with rnd_seeds: 42, 23, 89, 30, 12
    )
    parser.add_argument(
        '--max-seq-len', 
        type=int, 
        help='Number of genes to sample during training', 
        default = 1536
    )
    parser.add_argument(
        '--dropout', 
        type=float, 
        help='dropout value', 
        default = 0.2
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        help='learning rate', 
        default = 1e-4
    )
    parser.add_argument(
        '--schedule-interval-lr', 
        type=int, 
        help='schedule interval for lr', 
        default = 1
    )
    parser.add_argument(
        '--early-stop', 
        type=int, 
        help='number of epochs to stop early if loss does not decrease', 
        default = 10
    )
    parser.add_argument(
        '--log-interval', 
        type=int, 
        help='number of interval for which to log', 
        default = 100
    )
    parser.add_argument(
        '--pretrained-model-dir', 
        type=str, 
        help='directory of pretrained models are in', 
        default = 'models/'
    )
    parser.add_argument(
        '--outputs_dir', 
        type=str, 
        help='directory where model outputs and metrics are saved', 
        default = 'outputs/'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":    
       
    args = get_args()    
    set_seed(args.rnd_seed)
    device = args.device
    dataset_name = args.dataset
    model_type = args.model_type
    use_fast_transformer = True  # whether to use fast transformer
    amp = True
    
    # Location of pretrained scGPT model
    scgpt_pretrained_model_location = args.pretrained_model_dir + 'pretrained/scgpt'
    
    # Location where the model outputs will be saved to 
    save_dir = Path(args.outputs_dir + dataset_name + "/" + model_type + "/seed_" + str(args.rnd_seed) + "/")
    make_output_dirs(save_dir)
    
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    
    # Log running date
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # learned parameters to load from scGPT model architecture
    load_param_prefixs = [
        "encoder",
        "value_encoder",
        "transformer_encoder"
    ]
    
    # Load data
    pert_data = load_dataloader(args.dataset, args.batch_size, args.eval_batch_size, split = 'simulation')
    
    # Get the embedding types to include in the model training 
    embs_to_include = get_embs_to_include(args.model_type)
    
    # Get gene vocab and IDs
    vocab_file = Path(scgpt_pretrained_model_location) / "vocab.json"
    vocab, gene_ids, dataset_genes, gene2idx = match_genes_to_scgpt_vocab(vocab_file, pert_data, logger, SPECIAL_TOKENS)
    ntokens = len(vocab)  # size of vocabulary
    
    # Get GenePT embeddings to include
    genept_embs, genept_emb_type, genept_emb_dim, found_genes_genept = initialize_genept_embeddings(embs_to_include, dataset_genes, vocab, args.model_type, args.pretrained_model_dir)
    
    # Get GO embeddings to include
    go_embs_to_include, go_emb_type, go_emb_dim, found_genes_go = initialize_go_embeddings(embs_to_include, dataset_genes, vocab, args.model_type, args.pretrained_model_dir)
    
    model = scGenePT(
        ntoken=ntokens,
        d_model=EMBSIZE,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        nlayers_cls=N_LAYERS_CLS,
        n_cls=N_CLS,
        vocab=vocab,
        n_perturbagens=2,
        dropout=args.dropout,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=use_fast_transformer,
        embs_to_include = embs_to_include,
        genept_embs = genept_embs, 
        genept_emb_type = genept_emb_type, 
        genept_emb_size = genept_emb_dim,
        go_embs_to_include = go_embs_to_include,
        go_emb_type = go_emb_type,
        go_emb_size = go_emb_dim
    )
    
    # If we don't to include learned attention, it needs to be taken out of the weights that are being initialize
    if 'no_attention' in args.model_type:
        load_param_prefixs = [
            "encoder",
            "value_encoder",
        ] 
        
    # Load weights from pretrained_model
    model = load_pretrained_model(model, load_param_prefixs, False, Path(scgpt_pretrained_model_location) / "best_model.pt", device)  
    model.to(device)
    
    # Lr functions
    loss_fn = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_interval_lr, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Train model
    save_models_each_epoch = False
    best_model = train_model(model, pert_data, args.num_epochs, loss_fn, optimizer, scheduler, scaler, device, gene_ids, logger, INCLUDE_ZERO_GENE, amp, dataset_name, args.model_type, args.rnd_seed, args.max_seq_len, args.log_interval, args.early_stop, gene2idx, save_models_each_epoch, save_dir)
    
    # Save best model under model output directory
    print(f"Saving best model under {save_dir}/models/best_model.pt")
    torch.save(best_model.state_dict(), save_dir / "models/best_model.pt")
    
    # Evaluate best model on test data  
    print(f"Evaluating best model on test data:")
    test_metrics = compute_test_metrics(pert_data, model, 'test', save_dir, device, INCLUDE_ZERO_GENE, gene_ids)
    with open(save_dir / "metrics/test/test_metrics_detailed.json", "w") as outfile:
        outfile.write(json.dumps(test_metrics))