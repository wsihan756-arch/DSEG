<h2 align="center">✨DSEG-MVC:Dynamic Spurious Erasure and Global guidance Multi-View Clustering</h2>


<p align="center">
  <b>Sihan Wang<sup>1</sup>, Yandong Li<sup>1</sup>, Yipeng Shi<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>School of Software, Hebei Normal University, Shijiazhuang 050024, China<br>
</p>

<p align="center">
  <!-- Contact Badge -->
  <a href="wsihan756@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-wsihan756%40gmail.com-blue.svg" alt="Contact Author">
  </a>
</p>

<p align="center">
  🔥 Our work has been accepted by ICIC 2026<br>
</p>

## Overview🔍
<div>
    <img src="https://github.com/wsihan756-arch/DSEG/blob/main/DSEG.png" width="90%" height="90%">
</div>

**Figure 1. Framework of DSEG-MVC.**


## Datasets

DSEG-MVC is evaluated on six benchmark datasets:

| Dataset | Samples | Views | Clusters |
|---|---:|---:|---:|
| MNIST-USPS | 5,000 | 2 | 10 |
| Synthetic3d | 600 | 3 | 3 |
| Fashion | 10,000 | 3 | 10 |
| BDGP | 2,500 | 2 | 5 |
| Mfeat | 2,000 | 6 | 10 |
| Hdigit | 10,000 | 2 | 10 |

## Experimental Results

DSEG-MVC is compared with seven recent multi-view clustering methods: DMCAG, DFP-GNN, DCMVC, MVCAN, DIVIDE, DCG, and GDCN.

### Reported Performance of DSEG-MVC

| Dataset | ACC | NMI | ARI |
|---|---:|---:|---:|
| MNIST-USPS | 99.92 | 99.76 | 99.82 |
| Synthetic3d | 98.50 | 92.67 | 95.54 |
| Fashion | 98.63 | 96.71 | 97.02 |
| BDGP | 98.94 | 96.65 | 97.38 |
| Mfeat | 97.30 | 93.96 | 94.10 |
| Hdigit | 99.89 | 99.63 | 99.76 |

Across all six datasets and all three metrics, DSEG-MVC achieves the best reported results among the compared methods.

## Ablation Study

The paper reports that both major objective components are important:

| Variant | MNIST-USPS ACC | Fashion ACC | Mfeat ACC |
|---|---:|---:|---:|
| Without decorrelation loss | 58.88 | 76.49 | 78.05 |
| Without NCE loss | 84.26 | 94.50 | 91.55 |
| Full DSEG-MVC | 99.92 | 98.63 | 97.30 |

## Getting Started🚀

### Datasets
- We have only uploaded three datasets; the other three datasets are too large to upload and will need to be downloaded by the reader.
### Training and Evaluation
- To train the DSEG, run: `train.py`. The prediction results obtained using the K-Means algorithm.

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{dsegmvc,
  title  = {DSEG-MVC: Dynamic Spurious Erasure and Global Guidance Multi-View Clustering},
  author = {Wang, Sihan and Li, Yandong and Shi, Yipeng}
}
```
