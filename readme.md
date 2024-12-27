# ELISE
This is the official implementation of **ELISE** (Effective and Lightweight Representation Learning for Signed Bipartite Graphs). 
The paper is submitted to Neural Networks(Elsevier), and is under review:

* Effective and Lightweight Representation Learning for Signed Bipartite Graphs <br/>
  Gyeongmin Gu, Minseo Jeon, Hyun-Je Song, Jinhong Jung<br/>(under review)


## Overview
How can we effectively and efficiently learn node representations in signed bipartite graphs? A signed bipartite graph is a graph consisting of two nodes sets where nodes of different types are positively or negative connected, and it has been extensively used to model various real-world relationships such as e-commerce, peer review systems, etc. To analyze such a graph, previous studies have focused on designing methods for learning node representations using graph neural networks (GNNs). In particular, these methods insert edges between nodes of the same type based on balance theory, enabling them to leverage augmented structures in their learning. However, the existing methods rely on a naive message passing design, which is prone to over-smoothing and susceptible to noisy interactions in real-world graphs. Furthermore, they suffer from computational inefficiency due to their heavy design and the significant increase in the number of added edges.

In this paper, we propose ELISE, an effective and lightweight GNN-based approach for learning signed bipartite graphs. We first extend personalized propagation to a signed bipartite graph, incorporating signed edges during message passing. This extension adheres to balance theory without introducing additional edges, mitigating the over-smoothing issue and enhancing representation power. We then jointly learn node embeddings on a low-rank approximation of the signed bipartite graph, which reduces potential noise and emphasizes its global structure, further improving expressiveness without significant loss of efficiency. We encapsulate these ideas into ELISE, designing it to be lightweight, unlike the previous methods that add too many edges and cause inefficiency. Through extensive experiments on real-world signed bipartite graphs, we demonstrate that ELISE outperforms its competitors for predicting link signs while providing faster training and inference time.

## Prerequisites

The packages used in this repository are as follows:
```
python==3.12.3
numpy==1.26.4
pytorch==2.2.1
tqdm==4.66.4
loguru==0.7.2
fire==0.7.0
```

You can create a conda environment with these packages by typing the following command in your terminal:
```bash
conda env create --file environment.yml
conda activate ELISE
```

## Datasets 
We provide datasets used in the paper for reproducibility. 
You can find raw datasets in `./datasets` folder where the file's name is `${DATASET}.tsv`. 
The `${DATASET}` is one of `review`, `bonanza`,  `ml-1m`, and `amazon-dm`.
This file contains the list of signed edges where each line consists of a tuple of `(src, dst, sign)`.
The details of the datasets are provided in the following table:

| **Dataset**                                    | **$\|\mathcal{U}\|$** | **$\|\mathcal{V}\|$** | **$\|\mathcal{E}\|$** | **$\|\mathcal{E}^{+}\|$** | **$\|\mathcal{E}^{-}\|$** | **$p$(+)%** |
|:----------------------------------------------:|----------------------:|----------------------:|----------------------:|-------------------------:|-------------------------:|------------:|
| [Review]([https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html](https://github.com/huangjunjie-cs/SBGNN/tree/main/experiments-data))  |                   182 |                   304 |                 1,170 |                     464 |                     706 |        40.3 |
| [Bonanza]([https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html](https://github.com/huangjunjie-cs/SBGNN/tree/main/experiments-data))   |                 7,919 |                 1,973 |                36,543 |                  35,805 |                     738 |        98.0 |
| [ML-1m]([https://snap.stanford.edu/data/wiki-RfA.html](https://github.com/huangjunjie-cs/SBGNN/tree/main/experiments-data))                |                 6,040 |                 3,706 |             1,000,209 |                 836,478 |                 163,731 |        83.6 |
| [Amazon-DM]([http://konect.cc/networks/slashdot-zoo](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/))                  |                11,796 |                16,565 |               169,781 |                 165,777 |                   4,004 |        97.6 |
* $\|\mathcal{U}\|$: the number of type of node $\mathcal{U}$  
* $\|\mathcal{V}\|$: the number of type of node $\mathcal{V}$  
* $\|\mathcal{E}\|$: the number of edges  
* $\|\mathcal{E}^{+}\|$ and $\|\mathcal{E}^{-}\|$: the numbers of positive and negative edges, respectively  
* $p$(+): the ratio of positive edges  

## Demo
You can run the simple demo by typing the following command in your terminal:
```bash
cd src
bash python -m main --dataset review
```

## Options

| Option              | Description                                      | Default        |
| ------------------- | ------------------------------------------------ | -------------- |
| model               | model name                                       | elise          |
| dataset_name        | dataset name (review, bonanza, ml-1m, amazon-dm) | review         |
| seed                | random seed value                                | 600            |
| device              | device name                                      | cuda:0         |
| epochs              | number of epoch                                  | 200            |
| lr                  | learning rate of an optimizer                    | 0.005          |
| wdc                 | L2 regularization $\lambda_{\text{reg}}$         | 0.0001         |
| num_layer           | number $L$ of layer                              | 2              |
| num_decoder_layer s | number of classifier layer                       | 2              |
| c                   | ratio $c$ of personalized injection              | 0.45           |
| rank_ratio          | ratio $r$ of rank                                | 0.7            |
| input_dim           | model input feature dimension                    | 32             |
| decoder_input_dim   | decoder input feature dimension                  | 256            |
| split_ratio         | ratio of split of dataset for each phase         | [0.85,0.05,01] |
| dataset_shuffle     | check the shuffle                                | true           |
| optimizer           | optimizer name                                   | Adam           |

## Result of ELISE 

|**Dataset**|**AUC**|**Macro-F1**|**Micro-F1**|**Binary-F1**|
|:-:|:-:|:-:|:-:|:-:|
|**Review**|      |      |      |      |
|**Bonanza**|      |      |      |      |
|**ml-1m**|      |      |      |      |
|**Amazon-dm**|      |      |      |      |

All experiments are conducted on RTX A5000 (24GB) with cuda version 12.0, and the above results were produced with the random seed `seed=1`.

## Citation

```javascript

```

