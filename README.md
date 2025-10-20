# S²F-GNN
## S²F-GNN: A Self-supervised Robust Encrypted Traffic Classification Method Based on Dynamic Heterogeneous Graphs

We understand that validating the model architecture and training process is crucial for peer review.
Therefore, we provide:
* Complete model architecture(DTTmodel.py)
* Construction of Heterogeneous Graphs Samples(DTT-creat.py)
* Pre-training and fine-tuning scripts(pretraining.py\fintuning.py)
* Pre-trained model weights
* A preprocessed sample dataset(ISCX VPN-NonVPN(VPN))

The specific hyperparameters have been indicated in the paper.
## Datasets
ISCX VPN-NonVPN(VPN): 
[Download Dataset](https://www.unb.ca/cic/datasets/vpn.html)

The original dataset has been constructed into heterogeneous graphs:
[Download](https://drive.google.com/drive/folders/1A1fuuPnYrbzOm65ry9mRBhsYKIePBUpl?usp=drive_link)
Pre-trained model weights:
[Download](https://drive.google.com/drive/folders/1HRCQC0a82ATuBMUr_vSWYraB1e_-MJGv?usp=drive_link)



## Operation
Construction of Heterogeneous Graphs Samples: python DTT-creat.py

Pre-training: 
```bash
python pretraining.py
```

Fine-tuning: 
```bash
python fintuning.py
```

## E-mail
If you have any question, please feel free to contact us by e-mail (jiangtaozhai@nuist.edu.cn).
