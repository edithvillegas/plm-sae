import pandas as pd
import numpy as np
import re

# Utilities ====================================

def sanitize_filename(filename):
    # Only keep alphanumeric characters, hyphen, underscore, period, and space
    sanitized = re.sub(r'[^a-zA-Z0-9\-_\. ]', '_', filename)
    return sanitized

# Main features ========================================================================================== 
def extract_pairs(dataframe, feature_name):
    """
    Returns a dictionary {"protein": [ [start-stop], [start-stop] ] }
    """
    feature_dict = dict()

    #iterate over all proteins
    for i, row in dataframe.iterrows():
        #if not nan
        if isinstance(row[feature_name], str):
            #parse start-stops into pairs [start,stop]
            feature_strings = row[feature_name].split(",")
            feature_pairs = list( map( 
                                lambda x: x.split("-"), 
                                feature_strings 
                                ) )
            #add +1 aminoacid to correct for CLS token
            feature_pairs = [
                                [ int(element)+1 if element.isdigit() else element for element in pair]
                                for pair in feature_pairs
                            ]
            #save
            feature_dict[row.protein_name] = feature_pairs
            
    return feature_dict

def extract_sites(dataframe, feature_name):
    """
    Returns a dictionary {"protein": [ [site], [site] ] }
    """
    feature_dict = dict()

    #iterate over all proteins
    for i, row in dataframe.iterrows():
        #if not nan
        if isinstance(row[feature_name], str):
            #parse location
            feature_strings = row[feature_name].split(",")
            
            #add +1 aminoacid to correct for CLS token
            feature_strings = [int(element)+1 for element in feature_strings if element.isdigit()]
            #save
            feature_dict[row.protein_name] = feature_strings
            
    return feature_dict

def get_contiguous_ids(feature_dict):
    """
    get aminoacid id span that contains the features (helix, strand, turn, intramembrane region)
    """
    sel_ids = []
    for protein_name, pairs in feature_dict.items():
        for pair in pairs:
            try:
                sel_ids += [f"{protein_name}_{i}" for i in range(  int(pair[0]) , int(pair[1])+1  )]
            except:
                print(f"Error parsing contiguous ids at {protein_name}")

    return sel_ids

def get_paired_ids(feature_dict):
    """
    get aminoacid id pair, sometimes other pair is in another chain 
    (not in the protein) and the pair is "start-" or "-stop" instead of "start-stop"
    """
    sel_ids = []
    for protein_name, pairs in feature_dict.items():
        for pair in pairs:
            #loop, in case only one element
            sel_ids += [f"{protein_name}_{i}" for i in pair ]

    return sel_ids

def get_site_ids(feature_dict):
    """
    get single aminoacid sites
    """
    sel_ids = []
    for protein_name, sites in feature_dict.items():
        sel_ids += [f"{protein_name}_{i}" for i in sites ]

    return sel_ids

# Granular Features =========================================================================== 
def parse_granular_feature(string):
    #split entries 
    annot_list = string.split(">")
    #split name from start-stop
    parsed_list = []
    for entry in annot_list:
        name, locations = entry.split(":")
        #split name
        if ";" in name:
            base_name, detailed_name = name.split(";")
        else:
            base_name, detailed_name = name, ''
        #split pairs
        pairs = locations.split("-")
        parsed_list.append( [base_name, detailed_name, pairs] )

    return parsed_list

def extract_pairs2(dataframe, feature_name):
    feature_dict = dict()

    #iterate over all proteins
    for i, row in dataframe.iterrows():
        #if not nan
        if isinstance(row[feature_name], str):
            try:
                feature_dict[row.protein_name] = parse_granular_feature(row[feature_name])
            except:
                print(f"Skipping {i} at protein {row.protein_name} for {feature_name}", i)
            
    return feature_dict

def get_contiguous_ids2(feature_dict, name_to_check=None, full_name_to_check=None):
    #only one of two levels must be specified
    if full_name_to_check is not None:
        assert name_to_check is None

    #get aminoacid ids that contain the features
    sel_ids = []
    
    #iterate over proteins
    for protein_name, value_lists in feature_dict.items():
        #iterate over annotation entries in protein
        for value_list in value_lists:
            base_name, detailed_name, pairs = value_list
            
            full_name = f"{base_name};{detailed_name}"
            #compare only alpha numeric characters in full name
            if full_name_to_check is not None:
                full_name_is_equal = [letter for letter in full_name if letter.isalnum()] == [letter for letter in full_name_to_check if letter.isalnum()]
            else:
                full_name_is_equal = False
                
            #check if value should be added 
            if ((name_to_check is not None and base_name==name_to_check) #if level 1 labels are equal 
                or (full_name_to_check is not None and full_name_is_equal) #if level 1&2 labels are equal
                or (name_to_check is None and full_name_to_check is None)): #if no label was specified
                
                # get only numbers in the pair (handle start-stop or start-)
                pair_list = [int(i)+1 for i in pairs if i.isdigit()] #add +1 aminoacid to correct for CLS token
                assert ( len(pair_list) in [1,2] ) #start-stop or start-

                if len(pair_list)==2:
                    sel_ids += [f"{protein_name}_{i}" for i in range(  int(pair_list[0]) , int(pair_list[1])+1  )]
                elif len(pair_list)==1:
                    sel_ids += [f"{protein_name}_{pair_list[0]}"]
                
    return sel_ids

# Protein Level =========================================================== 
#go terms, reactome
def get_paired_df(dataframe, feature_name):
    """
    Flattens out the annotations for each protein in a dataframe into a 
    protein_name, feature_annotation 
    paired dataframe with repeated protein_names where needed.
    """
    protein_names_list = []
    feature_ids_list = []
    
    #iterate over proteins
    for i, protein in dataframe.iterrows():
        
        if isinstance(protein[feature_name], str):
            feature_ids = protein[feature_name].split(",")
            
            #iterate over reactome annotations
            for feature_id in feature_ids:
                protein_names_list += [protein.protein_name]
                feature_ids_list += [feature_id]
    
    paired_annotation = pd.DataFrame({
                            "protein": protein_names_list,
                            "feature_id": feature_ids_list,
                        })
    
    return paired_annotation