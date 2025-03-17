import os, sys
import json
import pickle
import psutil
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import vstack

from parsing_uniprot_labels import *

"""
Creates dictionaries with activation summaries for token level annotations,
from a set of calculated sprot embeddings. It does it in *batches*. 
"""
# Input variables ===================================================== 

#input
sae_name="esm2_t6_8M_UR50D_31"
embs_name="esm2_6_31"
embedding_path = #path to calculated SAE embeddings batches
databatch_path = #path to corresponding annotation batches
num_batches = len(os.listdir(embedding_path))

#output
output_path= #output directory
os.makedirs(output_path, exist_ok=True)

#params
min_counts=100 #minimum number of counts to consider an annotation type
threshold_values = [0.01, 0.1, 1] #thresholds to decide when a neuron is active

#skipping variables 
skip_feature_name="none"
skip_tag = False

#features list ============================================================== 

#feature lists 
contiguous_features = ["helix", "strand", "turn", "intramembrane region"]
paired_features = ["disulfide bond"]
location_features = ["active site", "glycosylation site"]

#total features
total_features = []

for feature_name in (
    contiguous_features +
    paired_features +
    location_features
):
    total_features.append( (feature_name, "simple feature", "level_0") )

# hyerarchical features 
#first level
for feature_name in ["transmembrane region", "zinc finger region", "topological domain", "region of interest", "binding site"]:
    #get names to check - first level
    names_to_check = features_df[feature_name].dropna().apply(
                                        # name; name detail: position-position
                                        lambda x: x.split(":")[0].split(";")[0]
                                        ).value_counts()
    names_to_check=names_to_check[names_to_check>min_counts]
    for name_to_check in names_to_check.index.values.tolist():
        total_features.append( (feature_name, name_to_check, "level_1") )

#second level
for feature_name in ["transmembrane region"]:
    #get names to check - first level
    names_to_check = features_df[feature_name].dropna().apply(
                                        # name; name detail: position-position
                                        lambda x: x.split(":")[0]
                                        ).value_counts()
    names_to_check=names_to_check[names_to_check>min_counts]
    for name_to_check in names_to_check.index.values.tolist():
        total_features.append( (feature_name, name_to_check, "level_2") )

# Loop over batches ========================================================= 
for i_batch in range(0,num_batches):
    # Load embeddings ---------------------------------------------
    embs = sparse.load_npz(f"{embedding_path}/{i_batch}.npz")
    print(f" Finished loading batch {i_batch} at {datetime.now()}")
    print(f"Memory used: {psutil.Process().memory_info().rss/(1024**2):.2f} MB") 

    # Load dataframe batches
    sel_sprot = pd.read_csv(f"{databatch_path}/{i_batch}.csv")
    #sel_sprot = batches_sprot[i_batch]
    sel_sprot.reset_index(inplace=True, drop=True)

    sel_features = features_df[features_df.protein_name.isin(sel_sprot.protein_name)]
    sel_features.reset_index(inplace=True, drop=True)
    #sel_features = batches_features[i_batch]
    
    # get the aminoacid ids --------------------------------------
    aminoacid_ids=[]
    for i, row in sel_sprot.iterrows():
        aminoacid_ids += [f"{row.protein_name}_{idx}" for idx in range(row.seq_length+2)]
    aminoacid_ids = np.array(aminoacid_ids)
    
    print("Finished aminoacid list")
    print(f"Memory used: {psutil.Process().memory_info().rss/(1024**2):.2f} MB") 

    assert embs.shape[0]==len(aminoacid_ids)

    #interpretability ------------------------------- 
    for feature_name, name_to_check, level in total_features:
        features_dict = dict()

        #check if skipping current feature
        if (
            feature_name.startswith(skip_feature_name) or 
            name_to_check.startswith(skip_feature_name)
            ):
            skip_tag = False #stop skipping when we reach this name
        if skip_tag: #otherwise, skip it
            print(f"Skipping {feature_name}, {name_to_check}, {i_batch}")
            continue

        #start processing 
        print(f"Processing {feature_name}, {name_to_check}, {i_batch}")

        #parse features and get aminoacid lists depending on feature type -----------------
        #1.parse feature string into {"protein_id": [ [start-stop], [start-stop]]}
        #2.get selected aminoacid ids

        if level=="level_0":
            if feature_name in contiguous_features:
                features_dict[feature_name] = extract_pairs(sel_features, feature_name)
                sel_ids = get_contiguous_ids(features_dict[feature_name])
                
            if feature_name in paired_features:
                features_dict[feature_name] = extract_pairs(sel_features, feature_name)
                sel_ids = get_paired_ids(features_dict[feature_name])
        
            if feature_name in location_features:
                features_dict[feature_name] = extract_sites(sel_features, feature_name)
                sel_ids = get_site_ids(features_dict[feature_name])

        elif level=="level_1":
            features_dict[feature_name] = extract_pairs2(sel_features, feature_name)
            sel_ids = get_contiguous_ids2(features_dict[feature_name], name_to_check=name_to_check)

        elif level=="level_2":
            features_dict[feature_name] = extract_pairs2(sel_features, feature_name)
            sel_ids = get_contiguous_ids2(features_dict[feature_name], full_name_to_check=name_to_check)
        # ---------------------------------------------------------------------------------
        
        #get indices
        mask = np.isin(aminoacid_ids, sel_ids)
        sel_indxs = np.where(mask)[0] #where are aminoacid_ids that are also present in sel_ids?
        neg_indxs = np.where(~mask)[0]

        #get percentages
        if level=="level_0":
            complete_feature_name = feature_name
        else:
            complete_feature_name = f"{feature_name} - {name_to_check}"

        result_values = dict()
        result_values[complete_feature_name] = {
            "positive_labels": len(sel_indxs),
            "negative_labels": len(neg_indxs),
            "activated_positive":{},
            "activated_negative":{},
        }
        
        for threshold in threshold_values:
            result_values[complete_feature_name]["activated_positive"][threshold] = (
                (embs[sel_indxs, :].tocsc() > threshold).sum(axis=0) )
            result_values[complete_feature_name]["activated_negative"][threshold] = (
                (embs[neg_indxs, :].tocsc() > threshold).sum(axis=0) )
            
        #save dictionary
        with open(f"{output_path}/{i_batch}_{ sanitize_filename(complete_feature_name) }.pkl", "wb") as f:
            pickle.dump(result_values, f)

        print("Finished processing feature")
        print(f"Memory used: {psutil.Process().memory_info().rss/(1024**2):.2f} MB") 

    del result_values, embs, aminoacid_ids, mask, sel_indxs, neg_indxs
    
