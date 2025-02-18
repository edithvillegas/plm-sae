"""
Adapted from Neel Nanda's 1L-Sparse-Autoencoder
https://github.com/neelnanda-io/1L-Sparse-Autoencoder
"""

import torch
from torch import nn

import os
import random
import tqdm
import pickle, json

import wandb

from .SAE_evaluation import evaluate_sae

def train_sae(sparse_autoencoder, train_dataloader, freq_dataloader, cfg, 
              test_sequences=None, train_sequences=None, test_sub_n=10, evaluation=False, middle_evaluation=False,
              inferencer=None, extractor=None, 
              reset_neuron_step=500, #increase to 2000, too often
              print_loss_step=50, wandb_log=True, 
              wandb_project="SAE-test", 
             ):
    '''
    Function to train a Sparse Autoencoder. 

    #Input Parameters:
    - sparse_autoencoder: the sparse autoencoder object
    - train_dataloader: dataloader object for the training dataset
    - freq_dataloader: dataloader object for the test dead neurons dataset
    - reset_neuron_step: after how many batches to revive dead neurons
    - print_loss_step: after how many batches to save loss

    #Output:
    List of dictionaries with losses 
    '''
    
    layer_name = cfg['layer']
    
    #store losses for curves
    losses_list=[]
    #init weights&biases
    if wandb_log:
        wandb.init(project=wandb_project, config=cfg)
        run_name=wandb.run.name
    else:
        run_name=sparse_autoencoder.get_version()
    
    #initialize optimizer
    sae_optimizer = torch.optim.Adam(sparse_autoencoder.parameters(), 
                                     lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    iterator_length = len(train_dataloader)
    #LOOP OVER BATCHES =====================================
    for i in tqdm.trange(iterator_length):
        #select data
        activations = next(iter(train_dataloader)) 
        activations = activations.to(cfg["device"])
        
        #SAE forward
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = sparse_autoencoder(activations)
        
        #optimizer step
        loss.backward()
        sparse_autoencoder.make_decoder_weights_and_grad_unit_norm()
        sae_optimizer.step()
        sae_optimizer.zero_grad()

    
        #save loss 
        loss_dict = {
            "loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(),
        }

        #every N batches do (log, revive):
        #log
        if (i) % print_loss_step == 0:
            if middle_evaluation==True:
                #evaluate SAE on test sequences
                l0_test, l1_test, l2_test, seq_rec_accuracy_test, kl_divergence_test, cross_entropy_increase_test = evaluate_sae(
                                                                                                                        test_sequences[0:test_sub_n],
                                                                                                                        sparse_autoencoder, 
                                                                                                                        inferencer, extractor, 
                                                                                                                        layer_name,
                                                                                                                        )
                #evaluate SAE on train subset
                l0_train, l1_train, l2_train, seq_rec_accuracy_train, kl_divergence_train, cross_entropy_increase_train = evaluate_sae(
                                                                                                                            train_sequences[0:test_sub_n],
                                                                                                                            sparse_autoencoder, 
                                                                                                                            inferencer, extractor, 
                                                                                                                            layer_name,
                                                                                                                            )                       
                
                #create loss dict 
                loss_dict = {
                    "loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(),
                    #test sequences
                    "l0_test": l0_test, "l1_test":l1_test, "l2_test":l2_test, "seq_rec_accuracy_test":seq_rec_accuracy_test, 
                    "kl_divergence_test":kl_divergence_test, "cross_entropy_increase_test":cross_entropy_increase_test,
                    #train sequences
                    "l0_train": l0_train, "l1_train":l1_train, "l2_train":l2_train, "seq_rec_accuracy_train":seq_rec_accuracy_train, 
                    "kl_divergence_train":kl_divergence_train, "cross_entropy_increase_train":cross_entropy_increase_train,
                }
            else:
                loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
                
            #log loss to plot
            losses_list.append(loss_dict)
            if wandb_log: wandb.log(loss_dict)

        #revive dead neurons
        if (i+1) % reset_neuron_step == 0:
            #revive neurons
            freqs = sparse_autoencoder.get_freqs(freq_dataloader, 50, device="cuda")
            to_be_reset = (freqs<10**(-5.5))
            print("Resetting neurons!", to_be_reset.sum())
            sparse_autoencoder.re_init(to_be_reset)

        #on last step, count dead neurons
        if i==iterator_length-1 and evaluation==True:
            #evaluate SAE
            l0, l1, l2, seq_rec_accuracy, kl_divergence, cross_entropy_increase = evaluate_sae(
                                                                                    test_sequences,
                                                                                    sparse_autoencoder, 
                                                                                    inferencer, extractor, 
                                                                                    layer_name,
                                                                                    )
            #dead neurons
            freqs = sparse_autoencoder.get_freqs(freq_dataloader, 50, device="cuda")
            dead = (freqs<10**(-5.5)).sum().item()
            #create dict 
            final_evaluation = {
                "loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(),
                "l0": l0, "l1":l1, "l2":l2, "seq_rec_accuracy":seq_rec_accuracy, 
                "kl_divergence":kl_divergence, "cross_entropy_increase":cross_entropy_increase,
                "dead_neurons":dead,
            }

        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, activations
        torch.cuda.empty_cache()
    
    #save SAE at the end of training
    sparse_autoencoder.save(name=run_name)
    if wandb_log: wandb.finish()
        
    #save losses to pickle
    loss_file=f"{cfg['SAVE_DIR']}/{run_name}_loss.pickle"
    with open(loss_file, "wb") as f:
        if evaluation==True:
            loss_to_save = [losses_list, final_evaluation]
        else:
            loss_to_save = losses_list
            
        pickle.dump(loss_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return losses_list