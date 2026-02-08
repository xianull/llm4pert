# settings for data prcocessing; these come from scGPT configuration
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = 0  # for padding values
PERT_PAD_ID = 2 

INCLUDE_ZERO_GENE = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective

# settings for the model
EMBSIZE = 512  # embedding dimension
D_HID = 512  # dimension of the feedforward network model in nn.TransformerEncoder
NLAYERS = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
NHEAD = 8  # number of heads in nn.MultiheadAttention
N_LAYERS_CLS= 3
N_CLS = 1

