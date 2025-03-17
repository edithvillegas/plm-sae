import os, sys
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import sparse

from torch.cuda import empty_cache

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from sae.SAE_methods import AutoEncoder #import sparse autoencoder from local definition

# #Import Local Functions --------------------------------------------------
import sys, os
sys.path.append('../../src')
from sae import SAE_methods, SAE_training

# Input parameters --------------------------------------------------------
num_batches = 100

#paths
output_path=f"results/sprot_embeddings/{short_name}_{version}_v2"
os.makedirs(f"{output_path}", exist_ok=True)
os.makedirs(f"{output_path}/embeddings", exist_ok=True)
os.makedirs(f"{output_path}/data", exist_ok=True)

# Script ================================================================

#load ESM2 model 
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = model.to("cuda")

#load SAE (GPU-only)
sparse_autoencoder = AutoEncoder.from_pretrained("evillegasgarcia/sae_esm2_6_l3")

#load data 
sprot = load_dataset("evillegasgarcia/swissprot-proteins")
#filter out long sequences (OOM)
sprot = sprot[sprot.seq_length < 500]

#setup to extract ESM2 embeddings
layer_name = "esm.encoder.layer.3.output"
#define hook
intermediate_embs = dict()
def hook(module, input, output):
    intermediate_embs[layer_name] = output
return hook
#attach hook
hook_handle = model.esm.encoder.layer[3].output.register_forward_hook(l3_hook)

#inference ---------------------------------------------------------------- 

batches = np.array_split(sprot, num_batches)

for i, batch in enumerate(batches):
    print(f"Starting batch {i} at {datetime.now()}", flush=True)
    sparse_matrix_list = []
    
    #iterate over protein database
    for j, row in batch.iterrows():
        print(i, j)

        #get protein properties
        protein_id = row.protein_name
        sequence = row.sequence
    
        #PLM Inference
        tokenized = tokenizer.encode(sequence, return_tensors="pt")
        tokenized = tokenized.to("cuda")
        outputs = model(tokenized)
        embeddings = intermediate_embs[layer_name][0]
        
        #SAE Inference
        _, _, sae_embeddings, _, _ = sparse_autoencoder(embeddings)
        sae_embeddings = sae_embeddings.cpu().detach().numpy()
    
        #save to sparse matrix
        sparse_matrix = sparse.lil_matrix( sae_embeddings )
        sparse_matrix_list.append( sparse_matrix )
        
        #clear memory
        del outputs
        del tokenized
        del embs
        del sae_embeddings
        del sparse_matrix
        empty_cache()

    #save embeddings
    sparse_matrix = sparse.vstack( sparse_matrix_list ).tocsr()
    sparse.save_npz(f"{output_path}/embeddings/{i}", sparse_matrix)
    #save dataframe batch
    batch.to_csv(f"{output_path}/data/{i}.csv")

