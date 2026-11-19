# Seabed-Net: A multi-task network for joint bathymetry and pixel-based seabed classification from remote sensing imagery in shallow waters

[![Elsevier Paper](https://img.shields.io/static/v1?label=Elsevier&message=Paper&color=FF6600)](https://www.sciencedirect.com/science/article/pii/S0924271625004460) [![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2510.19329) [![MagicBathy](https://img.shields.io/badge/MagicBathy-Project-red.svg)](https://www.magicbathy.eu)

This repository provides the research implementation accompanying the article:

> Panagiotis Agrafiotis and Begüm Demir (2026). **Seabed-Net: A multi-task network for joint bathymetry estimation and seabed classification from remote sensing imagery in shallow waters.** *ISPRS Journal of Photogrammetry and Remote Sensing*, **231**, 414–430. https://doi.org/10.1016/j.isprsjprs.2025.11.007.

## Availability and licensing

**Non-commercial research use.** The original research implementation is publicly available under the repository's [CC BY-NC-SA 4.0 licence](LICENSE.txt), subject to the terms and any third-party notices. Use, modification and redistribution permitted under that licence require appropriate attribution and compliance with its NonCommercial and ShareAlike conditions. The associated publication should be cited when using Seabed-Net in scientific work.

**Commercial licensing and implementation.** Commercial uses are not authorised by the public NonCommercial licence and require a separate written agreement with the applicable rights holder(s). For enquiries about commercial licensing, professional integration or operational deployment, contact **[3DeepVision P.C.](https://3deepvision.eu/)**. This contact designation does not change the original copyright ownership or scientific attribution.

**Research provenance.** Seabed-Net was developed in the context of the EU-funded MagicBathy research project, with work carried out at the Remote Sensing Image Analysis group of Technische Universität Berlin. 3DeepVision is the commercial contact for implementation enquiries; the research and publication remain attributed to their original authors and institutions.

**External assets.** Externally hosted pretrained models, third-party dependencies and the MagicBathyNet dataset may have additional or distinct terms. Review the notices applicable to each asset. Use of outputs for hydrographic, navigational or other safety-critical purposes requires independent validation; the research results do not constitute hydrographic certification.

## Abstract of the respective paper

Accurate, detailed, and regularly updated bathymetry, coupled with complex semantic content, is essential for under-mapped shallow-water environments facing increasing climatological and anthropogenic pressures. However, existing approaches that derive either depth or seabed classes from remote sensing imagery treat these tasks in isolation, forfeiting the mutual benefits of their interaction and hindering broader adoption of deep learning methods. To address these limitations, we introduce Seabed-Net, a unified multi-task framework that simultaneously predicts bathymetry and pixel-based seabed classification from remote sensing imagery of various resolutions. Seabed-Net employs dualbranch encoders for bathymetry estimation and pixel-based seabed classification, integrates cross-task features via an Attention Feature Fusion module and a windowed Swin-Transformer fusion block, and balances objectives through dynamic task uncertainty weighting. In extensive evaluations at two heterogeneous coastal sites, it reduces bathymetric RMSE by 10-30% compared to single-task and state-of-the-art multi-task baselines and improves seabed classification accuracy up to 8%. Qualitative analyses further demonstrate enhanced spatial consistency, sharper habitat boundaries, and corrected depth biases in low-contrast regions. These results confirm that jointly modeling depth with both substrate and seabed habitats yields synergistic gains, offering a robust, open solution for integrated shallow-water mapping.

## Citation

If you find this repository useful, please consider giving it a star ⭐.

If you use the code in this repository, please cite:

> Panagiotis Agrafiotis and Begüm Demir (2026). Seabed-Net: A multi-task network for joint bathymetry estimation and seabed classification from remote sensing imagery in shallow waters. *ISPRS Journal of Photogrammetry and Remote Sensing*, 231, 414–430. https://doi.org/10.1016/j.isprsjprs.2025.11.007.

```bibtex
@article{AGRAFIOTIS2026414,
  title = {Seabed-Net: A multi-task network for joint bathymetry estimation and seabed classification from remote sensing imagery in shallow waters},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {231},
  pages = {414-430},
  year = {2026},
  issn = {0924-2716},
  doi = {10.1016/j.isprsjprs.2025.11.007},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271625004460},
  author = {Panagiotis Agrafiotis and Begüm Demir},
  keywords = {Multi-task learning, Remote sensing, Shallow-water mapping, Seabed classification, Learning-based bathymetry, Deep learning}
}
```

If you use the MagicBathyNet dataset, please cite:

> P. Agrafiotis, Ł. Janowski, D. Skarlatos and B. Demir, "MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters," *IGARSS 2024*, Athens, Greece, 2024, pp. 249–253. https://doi.org/10.1109/IGARSS53475.2024.10641355.

```bibtex
@INPROCEEDINGS{10641355,
  author={Agrafiotis, Panagiotis and Janowski, Łukasz and Skarlatos, Dimitrios and Demir, Begüm},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium},
  title={MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters},
  year={2024},
  volume={},
  number={},
  pages={249-253},
  doi={10.1109/IGARSS53475.2024.10641355}
}
```

## Architecture overview

Seabed-Net jointly estimates bathymetric depth and performs pixel-level seabed classification from satellite and aerial remote sensing imagery. Unlike prior models that treat these tasks independently or use one as an auxiliary, Seabed-Net employs a multi-task framework where both outputs are supervised and contribute to shared representation learning. The architecture integrates spatially adaptive attention and windowed Transformer components to capture both local and global features.

<img width="2063" height="938" alt="Seabed-Net architecture" src="https://github.com/user-attachments/assets/d17de0a0-1d16-47c0-a5cb-11daa8be686f" />

## Getting started

### Downloading the dataset

For dataset downloads and documentation, visit the [MagicBathyNet project website](https://www.magicbathy.eu/magicbathynet.html). Review dataset-specific conditions independently of the code licence.

### Clone the repository

```bash
git clone https://github.com/pagraf/Seabed-Net.git
```

### Installation guide

The requirements can be installed with Anaconda (recommended), using the environment specification where provided:

```bash
conda env create -f environment.yml
conda activate magicbathynet
```

### Train and test the models

The original experimental workflow references **Swin-BathyUNet_MVS_SDB_Intergration_pub.ipynb**. Refer to the files available in the repository and any accompanying release materials for current training and testing scripts.

## Pretrained Seabed-Net models

Model weights are linked for the following modalities and regions. Review any applicable terms accompanying the hosted archives before reuse.

| Model | Pretrained PyTorch models |
| --- | --- |
| Seabed-Net – Agia Napa – Aerial | [Seabed-Net_Agia_Napa_Aerial.zip](https://drive.google.com/file/d/1PCpxUCybtARTBV1Yv2W3Fxvxk8kOGNRz/view?usp=sharing) |
| Seabed-Net – Agia Napa – SPOT 6 | [Seabed-Net_Agia_Napa_SPOT6.zip](https://drive.google.com/file/d/14EJWjRYFL8loZa3cd_PgpcIEAbnHQjGG/view?usp=sharing) |
| Seabed-Net – Agia Napa – Sentinel-2 | [Seabed-Net_Agia_Napa_S2.zip](https://drive.google.com/file/d/1VFfrhoQyNxqgxZeb-ZADaxucgwjnZA_b/view?usp=sharing) |
| Seabed-Net – Puck Lagoon – Aerial | [Seabed-Net_Puck_Lagoon_Aerial.zip](https://drive.google.com/file/d/1F2Ni6jaKlb0AOPWJX7V8ipVD3ca2uFz1/view?usp=sharing) |
| Seabed-Net – Puck Lagoon – SPOT 6 | [Seabed-Net_Puck_Lagoon_SPOT6.zip](https://drive.google.com/file/d/1P-F8f4KqsyjavMvvvIwkLlxuLdfDQyC2/view?usp=sharing) |
| Seabed-Net – Puck Lagoon – Sentinel-2 | [Seabed-Net_Puck_Lagoon_S2.zip](https://drive.google.com/file/d/13LMVHxhsMn_6DnH7G_jBbui-Jt_Ce3h3/view?usp=sharing) |

## Example testing results

![Bathymetry retrieval results](https://github.com/user-attachments/assets/529e3dfa-9ead-4570-adb6-b99e496a87e5)

Bathymetry retrieval results from aerial, SPOT 6 and Sentinel-2 data for Agia Napa, comparing (a) true-colour image, (b) UNet-Bathy, (c) PAD-Net, (d) MTI-Net, (e) MTL, (f) JSH-Net, (g) TaskPrompter, (h) Seabed-Net and (i) reference LiDAR/sonar data.

![Seabed classification results](https://github.com/user-attachments/assets/68f3de1e-3e2c-4bd7-962b-eb89875cfb84)

Pixel-based classification results from aerial, SPOT 6 and Sentinel-2 data for Agia Napa, comparing (a) true-colour image, (b) U-Net, (c) SegFormer, (d) PAD-Net, (e) MTI-Net, (f) MTL, (g) JSH-Net, (h) TaskPrompter and (i) Seabed-Net.

For further evaluation details and accuracy metrics, read the [research paper](https://arxiv.org/abs/2510.19329).

## Authors and contacts

**Scientific authors**

- [Panagiotis Agrafiotis](http://users.ntua.gr/pagraf/)
- Begüm Demir

**Scientific enquiries, reproducibility and research feedback:** pagraf@central.ntua.gr

**Commercial licensing, implementation and operational deployment enquiries:** [3DeepVision P.C.](https://3deepvision.eu/)

The scientific attribution of Seabed-Net remains with the paper's authors. The commercial contact above does not imply a transfer of copyright.

## Funding and acknowledgements

This research is part of the **MagicBathy project**, funded by the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement **No. 101063294**. The work was carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/) of Technische Universität Berlin. More information is available at [MagicBathy](https://www.magicbathy.eu/).
