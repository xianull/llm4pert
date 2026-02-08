
# scGenePT: Is language all you need for modeling single-cell perturbations?
## Model Description
scGenePT is a collection of single-cell models for perturbation prediction. It leverages the [scGPT](https://github.com/bowang-lab/scGPT) [1] foundation model for scRNAseq data by injecting language embeddings at the gene level into the model architecture. The language gene embeddings are obtained by embedding gene level information from different knowledge sources using LLMs. The knowledge sources used include NCBI gene descriptions, UniProt protein Summaries for protein coding genes - as inspired by the [genePT](https://github.com/yiqunchen/GenePT) [2] approach - and GO (Gene Ontology) Gene Molecular Annotations, across three different axes: Molecular Function, Biological Process and Cellular Component



## :file_folder: Data
All of the data - including pre-computed gene_embeddings, as well as trained models, can be found in the public s3 bucket:
```
s3://czi-scgenept-public/
```

The data can be accessed through the aws cli. In most cases, `pip install awscli` should provide the required functionality to download and see the files. For information on installing the aws cli, follow the [official documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

**To download a folder**: ```aws s3 sync --no-sign-request s3://czi-scgenept-public/models/finetuned/scgenept_go_c output_dir ```

**To download a file**: ```aws s3 sync --no-sign-request s3://czi-scgenept-public/models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle output_dir```

## scGenePT Model Zoo
Trained scGenePT Models can be downloaded from this Google Drive [link]()

Model | Description | Download link <br> aws s3 sync --no-sign-request [...]
--- | --- | ---
scgenept_ncbi | scGPT + NCBI Gene Card Summaries | s3://czi-scgenept-public/models/finetuned/scgenept_ncbi
scgenept_ncbi+uniprot | scGPT + NCBI Gene Card Summaries + UniProt Protein Summaries | s3://czi-scgenept-public/models/finetuned/scgenept_ncbi+uniprot
scgenept_go_c | scGPT + GO Cellular Components Annotations | s3://czi-scgenept-public/models/finetuned/scgenept_go_c
scgenept_go_f | scGPT + GO Molecular Functions Annotations | s3://czi-scgenept-public/models/finetuned/scgenept_go_f
scgenept_go_p | scGPT + GO Biological Processes Annotations | s3://czi-scgenept-public/models/finetuned/scgenept_go_p
scgenept_go_all | scGPT + GO_F + GO_C + GO_P | s3://czi-scgenept-public/models/finetuned/scgenept_go_all
scgpt | scGPT | s3://czi-scgenept-public/models/finetuned/scgpt

**scGPT Pretrained Model** <br>
Pretrained model | Download from | Should be under
---- | --- | --- 
scGPT Model weights (whole-human) | [scGPT Google Drive Link]() <br> s3://czi-scgenept-public/models/pretrained/scgpt | `models/pretrained/scgpt/` <br> - best_model.pt <br> - args.json <br> - vocab.json|


**Pre-Computed Gene Embeddings** <br>
All gene embeddings can be found under `s3://czi-scgenept-public/gene_embeddings/`. You can download all of them at once using
 ```aws s3 sync --no-sign-request s3://czi-scgenept-public/models/gene_embeddings gene_embeddings```

Gene Embedding | Download from <br> aws s3 sync --no-sign-request[...] | Should be under 
---- | ---- | --- |
NCBI Gene summaries | [GenePT zenodo Link](https://zenodo.org/records/10833191) <br> s3://czi-scgenept-public/gene_embeddings/ | `models/gene_embeddings/` <br> NCBI_gene_embeddings-gpt3.5-ada.pickle
NCBI Gene summaries + UniProt protein summaries| s3://czi-scgenept-public/models/gene_embeddings/| `models/gene_embeddings/` <br> NCBI+UniProt_embeddings-gpt3.5-ada.pkl
GO Cellular Components Annotations| s3://czi-scgenept-public/models/gene_embeddings/|`models/gene_embeddings/` <br> GO_C_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_C_gene_embeddings-gpt3.5-ada_avg.pickle
GO Molecular Function Annotations| s3://czi-scgenept-public/models/gene_embeddings/ |`models/gene_embeddings/` <br> GO_F_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_F_gene_embeddings-gpt3.5-ada_avg.pickle
GO Biological Processes Annotations| s3://czi-scgenept-public/models/gene_embeddings/| `models/gene_embeddings/` <br> GO_P_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_P_gene_embeddings-gpt3.5-ada_avg.pickle
Aggregation of GO-C + GO-F + GO-P| s3://czi-scgenept-public/models/gene_embeddings/|  `models/gene_embeddings/` <br> GO_all_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_all_gene_embeddings-gpt3.5-ada_avg.pickle

The **gene annotations** can be downloaded from `s3://czi-scgenept-public/models/gene_embeddings/gene_annotations`

## :chart_with_upwards_trend: Training 

**Step 1: Download pretrained scGPT model** <br>
```aws s3 sync --no-sign-request s3://czi-scgenept-public/models/pretrained/scgpt models/pretrained/```

**Step 2: Download pre-computed gene Embeddings** <br>

scGenePT can use multiple sources for textual gene annotations. The different sources and gene representations are described above, together with the download links. If you're only interested in using one type of gene embeddings, you only need to download those embeddings only. <br>

Example for training a model using the GO-C embeddings: ```aws s3 sync --no-sign-request s3://czi-scgenept-public/models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle```

**Step 3: Environment setup**

We highly recommend creating a virtual environment. Models have been trained using flash-attn. However, flash-attn installation might be finicky, in which case models can be trained without.
```
conda create -y --name scgenept python=3.10 # or python3.10 -m venv scgenept
source activate scgenept
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install scgpt "flash-attn<1.0.5"
```

**Step 4: Training Data** <br>

We use the processed versions of the Adamson and Norman datasets from [GEARS](https://github.com/snap-stanford/GEARS). Note that there are some differences in dataloaders between differrent versions, we trained and evaluated the models on GEARS v=0.0.2. This code snippet is already embedded in the the codebase, so no additional work is needed to train on these datasets. 

```python
from GEARS import PertData
dataset_name = 'norman' # or 'adamson'
pert_data = PertData("data/")
pert_data.load(data_name=dataset_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=val_batch_size)
```

**Step 5 Training Script** <br> ⚠️ Note that training requires a GPU

`python train.py --model-type=scgenept_ncbi+uniprot_gpt --num-epochs=20 --dataset=norman --device=cuda:0`

The model-type to train can be passed through the --model-type argument, which can be one of:

**scGenePT** | **genePT** | **scGenePT_combined**  | **scGPT**
---- | ---- | ---- | ----
scgenept_ncbi_gpt | genept_ncbi_gpt |  |
scgenept_ncbi+uniprot_gpt | genept_ncbi+uniprot_gpt | |
scgenept_go_c_gpt | go_c_gpt_concat | scgenept_ncbi+uniprot_gpt_go_c_gpt_concat |
scgenept_go_f_gpt | go_f_gpt_concat | scgenept_ncbi+uniprot_gpt_go_f_gpt_concat | scgpt
scgenept_go_p_gpt | go_p_gpt_concat | scgenept_ncbi+uniprot_gpt_go_p_gpt_concat | scgpt_counts
scgenept_go_all_gpt | go_all_gpt_concat | scgenept_ncbi+uniprot_gpt_go_all_gpt_concat | scgpt_tokens


More details on model_type can be found in the `get_embs_to_include(model_type)` function under `utils/data_loading.py`. For each of the model types, a suffix **_no_attention** can be added, which means that the model won't use scGPT pre-trained attention.
All other training parameters can be found in the script.

## :bar_chart: Inference

- [scgenept_tutorial](https://github.com/czi-ai/scGenePT/blob/main/tutorials/scgenept_tutorial.ipynb) - Tutorial showcasing how to use trained scGenePT models in inference mode for perturbation prediction. It uses models fine-tuned on the Norman dataset and offers examples of predicting post-perturbation expression responses for single and two-gene perturbations. <br>
For inference, we recommend not using flash attention: 
```
python3.10 -m venv scgenept
source scgenept/bin/activate
pip install -r requirements.txt
pip install scgpt 
```

Same tutorial can be found as a Google Collab notebook [here]()

## :bookmark: Cite Us
If you use scGenePT in your analyses, please cite us:

**Paper**: Istrate, Ana-Maria, Donghui Li, and Theofanis Karaletsos. "scGenePT: Is language all you need for modeling single-cell perturbations?." bioRxiv (2024): 2024-10. [bioRxiv Link](https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1)

```
@article{istrate2024scgenept,
  title={scGenePT: Is language all you need for modeling single-cell perturbations?},
  author={Istrate, Ana-Maria and Li, Donghui and Karaletsos, Theofanis},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## :star: Acknowledgements

We would like to sincerely thank the authors of the following models and packages:
- [scGPT](https://github.com/bowang-lab/scGPT)
- [GenePT](https://github.com/yiqunchen/GenePT)

## :paperclip: References
1. Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods (2024): 1-11. [Paper Link](https://www.nature.com/articles/s41592-024-02201-0) | [GitHub Repo](https://github.com/bowang-lab/scGPT) 
2. Chen, Yiqun, and James Zou. "GenePT: a simple but effective foundation model for genes and cells built from ChatGPT." bioRxiv (2024): 2023-10. [Paper Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824/) |  [GitHub Repo](https://github.com/yiqunchen/GenePT) 
5. Roohani, Yusuf, Kexin Huang, and Jure Leskovec. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." Nature Biotechnology 42.6 (2024): 927-935. [Paper Link](https://www.nature.com/articles/s41587-023-01905-6) | [GitHub Repo](https://github.com/snap-stanford/GEARS) 

