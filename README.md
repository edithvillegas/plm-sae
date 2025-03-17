# Protein Language Model - Sparse Autoencoders
❗ Repository under construction. Additional information coming soon!

This repository contains the accompanying code for the preprint [Interpreting and Steering Protein Language Models through Sparse Autoencoders](https://arxiv.org/abs/2502.09135) and instructions on how to use a Sparse Autoencoder trained on the ESM2-8M model. 

## Usage
### Extracting representations from the Sparse Autoencoder
The weights for the trained Sparse Autoencoder are available from huggingface [here](https://huggingface.co/evillegasgarcia/sae_esm2_6_l3).

- To use, first download the class defining the sparse autoencoder from github
```bash
git clone git@github.com:edithvillegas/plm-sae.git
cd plm-sae
```

- Load the base ESM2 model and the sparse autoencoder from huggingface.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sae.SAE_methods import AutoEncoder #import sparse autoencoder from local definition

#load ESM2 model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = model.to("cuda")

#load SAE (GPU-only)
sparse_autoencoder = AutoEncoder.from_pretrained("evillegasgarcia/sae_esm2_6_l3")
```

- Prepare auxiliary functions to extract embeddings from a specific point in the ESM2 model.

```python
#setup to extract ESM2 embeddings
layer_name = "esm.encoder.layer.3.output"
#define hook
intermediate_embs = dict()
def hook(module, input, output):
    intermediate_embs[layer_name] = output
return hook
#attach hook
hook_handle = model.esm.encoder.layer[3].output.register_forward_hook(l3_hook)
```

- Extract embeddings from the ESM2 model and then from the sparse autoencoder.

```python
#Inference
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPL"

#PLM Inference
tokenized = tokenizer.encode(sequence, return_tensors="pt")
tokenized = tokenized.to("cuda")
outputs = model(tokenized)
embeddings = intermediate_embs[layer_name][0]

#SAE Inference
_, _, sae_embeddings, _, _ = sparse_autoencoder(embeddings)
```


## Citations
If you use this work, please cite:
```bibtex
@misc{garcia2025interpretingsteeringproteinlanguage,
      title={Interpreting and Steering Protein Language Models through Sparse Autoencoders}, 
      author={Edith Natalia Villegas Garcia and Alessio Ansuini},
      year={2025},
      eprint={2502.09135},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.09135}, 
}
