# S²F-GNN
## S²F-GNN: A Self-supervised Robust Encrypted Traffic Classification Method Based on Dynamic Heterogeneous Graphs

We understand that validating the model architecture and training process is crucial for peer review.
Therefore, we provide:
1. Complete model architecture(DTTmodel.py)
2. Construction of Heterogeneous Graphs Samples(DTT-creat.py)
3. Pre-training and fine-tuning scripts(pretraining.py\fintuning.py)
4. Pre-trained model weights
5. A preprocessed sample dataset(ISCX VPN-NonVPN(VPN))

The specific hyperparameters have been indicated in the paper.

ISCX VPN-NonVPN(VPN): https://www.unb.ca/cic/datasets/vpn.html

The original dataset has been constructed into heterogeneous graphs:https://drive.google.com/drive/folders/1A1fuuPnYrbzOm65ry9mRBhsYKIePBUpl?usp=drive_link

## Operation
Construction of Heterogeneous Graphs Samples: python DTT-creat.py

Pre-training: python pretraining.py

Fine-tuning: python fintuning.py

## E-mail
If you have any question, please feel free to contact us by e-mail (jiangtaozhai@nuist.edu.cn).
