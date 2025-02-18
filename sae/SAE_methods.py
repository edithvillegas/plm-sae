"""
Adapted from Neel Nanda's 1L-Sparse-Autoencoder
https://github.com/neelnanda-io/1L-Sparse-Autoencoder
"""

import torch
from torch import nn
import numpy as np

import os
import random
import tqdm
import json

#configuration parameteres =================================================
def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    #cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    #cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Autoencoder Class ========================================================
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.device=cfg["device"]
        self.to(self.device)

        #loss type
        try:
            self.loss_type = cfg['loss_type']
        except:
            print("Loss not defined in config, using default l1 loss")
            self.loss_type = "l1"
        
        #save to
        self.SAVE_DIR = cfg['SAVE_DIR']
        self.cfg = cfg
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = torch.nn.functional.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        if self.loss_type=="l1":
            l1_loss = self.l1_coeff * (acts.float().abs().sum())
        elif self.loss_type=="hoyer":
            ratio = (acts.float().abs().sum())/(torch.sqrt(acts.float().pow(2).sum()))
            n_root = np.sqrt(self.d_hidden)
            sparsity = (n_root - ratio)/(n_root-1)
            l1_loss = (-1)*self.l1_coeff * sparsity
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    def encode(self, x):
        x_cent = x - self.b_dec
        acts = torch.nn.functional.relu(x_cent @ self.W_enc + self.b_enc)
        return acts

    def decode(self, acts):
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        '''
        Get consecutive version in folder.
        '''
        version_list = [int(file.split(".")[0]) for file in os.listdir(self.SAVE_DIR) if "pt" in str(file)] #🌸
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self, name):
        #save torch checkpoints
        torch.save(self.state_dict(), f"{self.SAVE_DIR}/{name}.pt")
        #save config file
        with open(f"{self.SAVE_DIR}/{name}_cfg.json", "w") as f:
            json.dump(self.cfg, f)
            
        print("Saved as version", name)
    
    @classmethod
    def load(cls, SAVE_DIR, version):
        cfg = (json.load(open(SAVE_DIR+"/"+(str(version)+"_cfg.json"), "r")))
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR+"/"+(str(version)+".pt")))
        return self

    # @classmethod
    # def load_from_hf(cls, version):
    #     """
    #     Loads the saved autoencoder from HuggingFace. 
    #     """
    #     cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
    #     return self

    # Dead Neuron Handling ===========================================
    @torch.no_grad()
    def get_freqs(self, dataloader, num_batches=25, device="cuda"):
        '''
        Get frequency of activations for neurons, to locate dead ones.
        '''
        local_encoder = self
        act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(device)
        total = 0
        for i in tqdm.trange(num_batches):
            acts = next(iter(dataloader)) #get data here
            acts = acts.to(device)
            #apply SAE
            hidden = local_encoder(acts)[2]
            #calculate frequency of activations
            act_freq_scores += (hidden > 0).sum(0)
            total+=hidden.shape[0]
        act_freq_scores /= total
        num_dead = (act_freq_scores==0).float().mean()
        print("Num dead", num_dead)
        return act_freq_scores
    
    
    @torch.no_grad()
    def re_init(self, indices):
        '''
        Re-initialize dead neurons
        '''
        encoder=self
        new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
        new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
        new_b_enc = (torch.zeros_like(encoder.b_enc))
        #print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
        encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
        encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
        encoder.b_enc.data[indices] = new_b_enc[indices]
