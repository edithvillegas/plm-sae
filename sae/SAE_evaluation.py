import numpy as np
import torch

from scipy.stats import entropy


def create_sae_hook(sparse_autoencoder):
    ''' 
    Create a hook that passes activations through a SAE.
    '''
    def sae_hook(module, input, output):
        #modify activations with SAE
        loss, x_reconstruct, acts, l2_loss, l1_loss = sparse_autoencoder(output[0])
        #reformat output to tuple
        output = (x_reconstruct,)
        return output
    #return hook
    return sae_hook

def evaluate_sae(
    sequence_list,
    sparse_autoencoder, inferencer, extractor, layer_name,
):

    #cross entropy (input, target)
    ce_loss = torch.nn.CrossEntropyLoss()
    
    #define hook
    sae_hook = create_sae_hook(sparse_autoencoder)

    #hooked module
    for name, named_module in inferencer.model.named_modules():
        if layer_name == name:
            module = named_module
    
    #define metrics
    l0, l1, l2 = 0, 0, 0
    seq_rec_accuracy = 0
    kl_divergence = 0
    cross_entropy_increase = 0
    
    for sequence in sequence_list:
        # baseline ESM output
        tokenized = inferencer.process(sequence.split() )
        outputs = inferencer.inference(tokenized)
        outputs = outputs['logits'][0].cpu().detach()
        # baseline probabilities
        base_p = torch.nn.functional.softmax(outputs, dim=1)
        base_p = base_p.numpy()
        #ESM embs
        embs = extractor.intermediate_outputs[layer_name][0]
        
        #sparse autoencoder
        loss, x_r, acts, l2_loss, l1_loss = sparse_autoencoder(embs) 
        #MSE
        l2_loss = l2_loss.mean()
        #l0 - number of alive neurons per token
        l0_loss = torch.nonzero(acts[0]).shape[0]/(acts.shape[1]) 
    
        #compare -----------------------------------------
        #add SAE hook
        module = inferencer.model.esm.encoder.layer[3]
        sae_handle = module.register_forward_hook(sae_hook) 
        
        # SAE output
        tokenized = inferencer.process(sequence.split() )
        outputs_saed = inferencer.inference(tokenized)
        outputs_saed = outputs_saed['logits'][0].cpu().detach()
        saed_p = torch.nn.functional.softmax(outputs_saed, dim=1)
        saed_p = saed_p.numpy()
        
        #sequence reconstruction accuracy
        seq_acc = sum( outputs.argmax(axis=1)==outputs_saed.argmax(axis=1) )/len(sequence)
        
        #KL divergence
        kl_div = entropy(base_p, saed_p, axis=1).mean()

        #CE differene
        ce_delta = ce_loss(outputs_saed, tokenized['input_ids'][0]) - ce_loss(outputs, tokenized['input_ids'][0]) #input, target
        
        #remove SAE hook
        sae_handle.remove()
        # ---------------------------------------------------
        
        #add results
        l0 += l0_loss
        l1 += l1_loss.detach().cpu().numpy()
        l2 += l2_loss.detach().cpu().numpy()
        seq_rec_accuracy += seq_acc.numpy()
        kl_divergence += kl_div
        cross_entropy_increase += ce_delta.numpy()
    
        #delete outputs
        del loss, x_r, acts, l2_loss, l1_loss
        del tokenized, outputs, outputs_saed, embs
        del base_p, saed_p
        
    #normalize 
    sequence_number = len(sequence_list)
    
    l0 /= sequence_number
    l1 /= sequence_number
    l2 /= sequence_number
    seq_rec_accuracy /= sequence_number
    kl_divergence /= sequence_number
    cross_entropy_increase /= sequence_number

    return l0, l1, l2, seq_rec_accuracy, kl_divergence, cross_entropy_increase