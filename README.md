# Seabed-Net: A multi-task network for joint bathymetry and pixel-based seabed classification from remote sensing imagery in shallow waters

This repository contains the code of the paper 
>Panagiotis Agrafiotis, Begüm Demir,
Seabed-Net: A multi-task network for joint bathymetry and pixel-based seabed classification from remote sensing imagery in shallow waters,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume XXX,
2025,
Pages XXX-XXX,
ISSN 0924-2716
<br />

## Abstract of the respective paper [![Elsevier Paper](https://img.shields.io/static/v1?label=Elsevier&message=Paper&color=FF6600)](https://doi.org/10.1016/j.isprsjprs.2025.04.020) [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.11416) [![MagicBathy](https://img.shields.io/badge/MagicBathy-Project-red.svg)](https://www.magicbathy.eu)
Accurate, detailed, and regularly updated bathymetry, coupled with complex semantic content, is essential for under-mapped shallow-water environments facing increasing climatological and anthropogenic pressures. However, existing approaches that derive either depth or seabed classes from remote sensing imagery treat these tasks in isolation, forfeiting the mutual benefits of their interaction and hindering broader adoption of deep learning methods. To address these limitations, we introduce Seabed-Net, a unified multi-task framework that simultaneously predicts bathymetry and pixel-based seabed classification from remote sensing imagery of various resolutions. Seabed-Net employs dualbranch encoders for bathymetry estimation and pixel-based seabed classification, integrates cross-task features via an Attention Feature Fusion module and a windowed Swin-Transformer fusion block, and balances objectives through dynamic task uncertainty weighting. In extensive evaluations at two heterogeneous coastal sites, it reduces bathymetric RMSE by 10-30% compared to single-task and state-of-the-art multi-task baselines and improves seabed classification accuracy up to 8%. Qualitative analyses further demonstrate enhanced spatial consistency, sharper habitat boundaries, and corrected depth biases in low-contrast regions. These results confirm that jointly modeling depth with both substrate and seabed habitats yields synergistic gains, offering a robust, open solution for integrated shallow-water mapping. Code and pretrained models will be made available upon acceptance.


## Citation

If you find this repository useful, please consider giving a star ⭐.
<br />

If you use the code in this repository please cite:

>Panagiotis Agrafiotis, Begüm Demir,
Seabed-Net: A multi-task network for joint bathymetry and pixel-based seabed classification from remote sensing imagery in shallow waters,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume XXX,
2025,
Pages XXX-XXX,
ISSN 0924-2716
<br />


```
@article{AGRAFIOTIS2025341,
title = {Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {225},
pages = {341-361},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.04.020},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001522},
author = {Panagiotis Agrafiotis and Begüm Demir},
}
```

If you use the dataset please cite:

>P. Agrafiotis, Ł. Janowski, D. Skarlatos and B. Demir, "MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters," IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, Athens, Greece, 2024, pp. 249-253, doi: 10.1109/IGARSS53475.2024.10641355.
```
@INPROCEEDINGS{10641355,
  author={Agrafiotis, Panagiotis and Janowski, Łukasz and Skarlatos, Dimitrios and Demir, Begüm},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters}, 
  year={2024},
  volume={},
  number={},
  pages={249-253},
  doi={10.1109/IGARSS53475.2024.10641355}}
```

# Architecture Overview
Seabed-NeT jointly estimates bathymetric depth and performs pixel-level seabed classification from satellite and aerial remote sensing imagery. Unlike prior models that treat these tasks independently or use one as an auxiliary, Seabed-Net employs a multi-task framework, where both outputs are supervised and contribute to shared representation learning. The architecture integrates spatially adaptive attention (FAA) and Vision Transformer (ViT) components, enabling it to capture both local and global features across diverse sensing modalities.

![seabed-net](https://github.com/user-attachments/assets/eacca1dc-60aa-4c15-a525-ab4d98600a08)


# Getting started

## Downloading the dataset

For downloading the dataset and a detailed explanation of it  (in case you don't have your own data or wish to compare), please visit the MagicBathy Project website at [https://www.magicbathy.eu/magicbathynet.html](https://www.magicbathy.eu/magicbathynet.html)

## Clone the repo

`git clone https://github.com/pagraf/Seabed-Net.git`

## Installation Guide
The requirements are easily installed via Anaconda (recommended):

`conda env create -f environment.yml`

After the installation is completed, activate the environment:

`conda activate magicbathynet`

## Train and Test the models
To train and test the model use **Swin-BathyUNet_MVS_SDB_Intergration_pub.ipynb**.


## Pre-trained Seabed-Net Models
We provide model weights for the following modalities and areas:

| Model Names | Pre-Trained PyTorch Models                                                                                | 
| ----------- |-----------------------------------------------------------------------------------------------------------|
| Seabed-Net - Agia Napa - Aerial | [NDR-Restore.zip](https://drive.google.com/file/d/1PCpxUCybtARTBV1Yv2W3Fxvxk8kOGNRz/view?usp=sharing)  |
| Seabed-Net - Agia Napa - SPOT 6 | [NDR-Restore.zip](https://drive.google.com/file/d/14EJWjRYFL8loZa3cd_PgpcIEAbnHQjGG/view?usp=sharing)  |
| Seabed-Net - Agia Napa - S2 | [NDR-Restore.zip](https://drive.google.com/file/d/1VFfrhoQyNxqgxZeb-ZADaxucgwjnZA_b/view?usp=sharing)  |
| Seabed-Net - Puck Lagoon - Aerial | [NDR-Restore.zip](https://drive.google.com/file/d/1F2Ni6jaKlb0AOPWJX7V8ipVD3ca2uFz1/view?usp=sharing)  |
| Seabed-Net - Puck Lagoon - SPOT 6 | [NDR-Restore.zip](https://drive.google.com/file/d/1P-F8f4KqsyjavMvvvIwkLlxuLdfDQyC2/view?usp=sharing)  |
| Seabed-Net - Puck Lagoon - S2 | [NDR-Restore.zip](https://drive.google.com/file/d/13LMVHxhsMn_6DnH7G_jBbui-Jt_Ce3h3/view?usp=sharing)  |
 
## Example testing results
![figure5](https://github.com/user-attachments/assets/529e3dfa-9ead-4570-adb6-b99e496a87e5)
Bathymetry retrieval results on the aerial, SPOT 6 and Sentinel-2 modalities from the compared single-task and
multi-task approaches. (a) True color composite of example patches acquired over Agia Napa, bathymetry obtained by (b)
UNet-Bathy, (c) PAD-Net, (d) MTI-Net, (e) MTL, (f) JSH-Net, (g) TaskPrompter, (h) Seabed-Net and (i) LiDAR/SONAR.

![figure7](https://github.com/user-attachments/assets/68f3de1e-3e2c-4bd7-962b-eb89875cfb84)
Pixel-based classification results on the aerial, SPOT 6 and Sentinel-2 modalities from the compared single-task and
multi-task approaches. (a) True color composite of example patches acquired over Agia Napa, seabed classes obtained by (b)
U-Net, (c) SegFormer, (d) PAD-Net, (e) MTI-Net, (f) MTL, (g) JSH-Net, (h) TaskPrompter, and (i) Seabed-Net. 


For more information on the results and accuracy achieved read our [paper](https://doi.org/10.1016/j.isprsjprs.2025.04.020). 

## Authors
Panagiotis Agrafiotis [https://www.user.tu-berlin.de/pagraf/](https://www.user.tu-berlin.de/pagraf/)

## Feedback
Feel free to give feedback, by sending an email to: agrafiotis@tu-berlin.de
<br />
<br />

# Funding
This work is part of **MagicBathy project funded by the European Union’s HORIZON Europe research and innovation programme under the Marie Skłodowska-Curie GA 101063294**. Work has been carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/). For more information about the project visit [https://www.magicbathy.eu/](https://www.magicbathy.eu/).
