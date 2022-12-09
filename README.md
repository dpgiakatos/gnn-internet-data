# gnn-internet-data

This repository contains material related to the paper 

`Dimitrios P. Giakatos, Sofia Kostoglou, Pavlos Sermpezis, Athena Vakali, "Benchmarking Graph Neural Networks for Internet Routing Data", in Proc. ACM CoNEXT (GNNet workshop), December 2022`

 

Paper link: [arXiv](https://arxiv.org/pdf/2210.14189.pdf) and [ACM DL](https://dl.acm.org/doi/10.1145/3565473.3569187)

Paper presentation: [slides](./GNNet2022_Benchmarking_GNNs.pdf)


### Prerequisites
* CUDA Version: 11.3
* Python version: 3.9

### Installation
```commandline
pip install -r requirements.txt
```

### Run benchmarks
- Link prediction
```commandline
python link-prediction.py
```
- Node classification
```commandline
python node-classification.py
```
