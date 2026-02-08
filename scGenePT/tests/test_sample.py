from utils.data_loading import *

def test_get_embs_to_include():
    """
    Tests if the right embeddings get included based on model name provided during cmd line args
    """
    assert(set(get_embs_to_include('scgpt')) == {'scGPT_counts_embs', 'scGPT_token_embs'})
    assert(set(get_embs_to_include('scgenept_ncbi_gpt')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'genePT_token_embs_gpt'})
    assert(set(get_embs_to_include('scgenept_ncbi+uniprot_gpt')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'genePT_token_embs_gpt'})
    assert(set(get_embs_to_include('scgenept_go_c_gpt_concat')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'GO_token_embs_gpt_concat'})
    assert(set(get_embs_to_include('scgenept_go_f_gpt_concat')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'GO_token_embs_gpt_concat'})
    assert(set(get_embs_to_include('scgenept_go_p_gpt_concat')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'GO_token_embs_gpt_concat'})
    assert(set(get_embs_to_include('scgenept_go_all_gpt_concat')) == {'scGPT_counts_embs', 'scGPT_token_embs', 'GO_token_embs_gpt_concat'})
 