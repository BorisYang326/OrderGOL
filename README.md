<div align="center">
<h1>Order Matters: Learning Element Ordering for <br> Graphic Design Generation (TOG 2025)</h1>

[Bo Yang](https://borisyang326.github.io/), [Ying Cao†](https://www.ying-cao.com/)

<sup>†</sup>Corresponding Author

ShanghaiTech University

<a href='https://dl.acm.org/doi/10.1145/3730858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='https://borisyang326.github.io/ordermatters.html'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://github.com/borisyang326/OrderGOL'><img src='https://img.shields.io/badge/Github-Code-bb8a2e?logo=github'></a>
[![GitHub](https://img.shields.io/github/stars/borisyang326/OrderGOL?style=social)](https://github.com/borisyang326/OrderGOL)

</div>

<p align="center">
  <img src="assets/teaser_v5.jpg" alt="Teaser showing comparison of graphic designs generated with different ordering strategies">
</p>

## 🏷️ Change Log 

- [2025/08/05] 🔥 We release the source code.
- [2025/04/30] 🔥 We release the <a href='https://borisyang326.github.io/ordermatters.html'>project page</a>.
- [2025/04/27] 📄 Paper accepted to ACM Transactions on Graphics (SIGGRAPH 2025).

## 🔆 Method Overview

<p align="center">
  <img src="assets/full_pipe-v4.png" alt="GOL Framework Pipeline">
</p>

We propose a **G**enerative **O**rder **L**earner (GOL) that learns optimal element ordering strategies for graphic design generation. Our approach consists of:

- **Design Generator**: A Transformer-based autoregressive model that generates design sequences
- **Ordering Network**: A neural network that learns content-adaptive element ordering
- **Joint Training**: Both networks are trained simultaneously for mutual improvement

The key insight is that the order in which design elements are generated significantly impacts the quality of the final design. Our neural order outperforms random and raster ordering strategies.

## ⚙️ Setup
```
conda env create -f requirements.yml
conda activate ordergol   
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
## 📊 Dataset Preparation

## 🚀 Training

## 🎯 Inference

## 📁 Project Structure

<!-- ```
OrderGOL/
├── configs/                 # Configuration files
├── data/                   # Dataset directory
├── checkpoints/            # Model checkpoints
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── scripts/               # Training and evaluation scripts
├── assets/                # Images and figures
├── requirements.txt       # Python dependencies
└── README.md             # This file
``` -->

## ❤️ Acknowledgments

This project is built upon several excellent open-source projects:
- [LayoutDM](https://github.com/CyberAgentAILab/layout-dm) LayoutDM: Discrete Diffusion Model for Controllable Layout Generation
- [LayoutTrans](https://github.com/kampta/DeepLayout) LayoutTransformer: Layout Generation and Completion with Self-attention

We thank the authors of these projects for their contributions to the open-source community.

## 🎓 Citation

If our work helps your research or applications, please consider citing our paper:

```bibtex
@article{yang2025order,
  author = {Yang, Bo and Cao, Ying},
  title = {Order Matters: Learning Element Ordering for Graphic Design Generation},
  year = {2025},
  issue_date = {August 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {44},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3730858},
  doi = {10.1145/3730858},
  journal = {ACM Trans. Graph.},
  articleno = {34},
  numpages = {16}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
