from utils.data_loading import *
from utils.scgpt_config import *
from utils.evaluation import *

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
        '--model-type', 
        type=str, 
        help='Type of model to train. One of: [scgpt, scgenept_ncbi_gpt, scgenept_ncbi+uniprot_gpt, scgenept_go_c_gpt_concat, scgenept_go_f_gpt_concat, scgenept_go_p_gpt_concat, scgenept_go_all_gpt_concat, genept_ncbi_gpt, genept_ncbi+uniprot_gpt, go_c_gpt_concat, go_f_gpt_concat, go_p_gpt_concat, go_all_gpt_concat ...]. For full list of possible models, please visit https://github.com/czi-ai/scGenePT.', 
        default = "scgpt"
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
        '--rnd-seed', 
        type=int, 
        help='random seed', 
        default = 42
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='dataset to train on; ', 
        default = 'adamson'
    )
    parser.add_argument(
        '--trained-model-dir', 
        type=str, 
        help='directory of pretrained models are in', 
        default = 'models/finetuned'
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
    device = args.device
    dataset_name = args.dataset
    model_type = args.model_type
    use_fast_transformer = True  # whether to use fast transformer
    amp = True
    
    # Location of pretrained scGPT model
    if args.model_type != 'scgpt':
        dir_model = args.model_type.split('_gpt')[0]
        trained_model_location = f'{args.trained_model_dir}/{dir_model}/{args.dataset}/'
        
        # Note that these are the extensions that will be available by default if you downloaded
        # all the models from AWS s3 bucket. If they models were renamed, these suffixes need to be changed
        if args.model_type == 'scgenept_ncbi+uniprot_gpt' and args.dataset == 'norman':
            trained_model_location += 'best_model_gpt3.5_ada_rnd_seed_42.pt'
        else:
            trained_model_location += 'best_model_gpt3.5_ada_rnd_seed_42_concat.pt'
    else:
        trained_model_location = f'{args.trained_model_dir}/{args.model_type}/{args.dataset}/'
        trained_model_location += 'best_model_seed_42.pt'
    
    # Location where the model outputs will be saved to 
    save_dir = Path(args.outputs_dir + dataset_name + "/" + model_type + "/seed_" + str(args.rnd_seed) + "/")
    make_output_dirs(save_dir)
    
    # Load data
    pert_data = load_dataloader(args.dataset, args.batch_size, args.eval_batch_size, split = 'simulation')
    pert_adata = pert_data.adata
    
    model, gene_ids =  load_trained_scgenept_model(pert_adata, model_type, 'models/', trained_model_location, device, verbose = False)
    model.to(device)
    print(model)
   
    print(f"Loaded best model from {trained_model_location}")
    
    # Evaluate best model on test data  
    print(f"Evaluating best model on test data:")
    test_metrics = compute_test_metrics(pert_data, model, 'test', save_dir, device, INCLUDE_ZERO_GENE, gene_ids)
    with open(save_dir / "metrics/test/test_metrics_detailed.json", "w") as outfile:
        outfile.write(json.dumps(test_metrics))