# Self-Supervised Learning for Annotation Efficient Biomedical Image Segmentation

This is the implementation of our puplication in IEEE Transactions on Biomedical Engineering. This work is implemeneted in [PyTorch](https://pytorch.org/) and more specifically [PyTorch Lightning](https://www.pytorchlightning.ai/).


>*Objective*: The scarcity of high-quality annotated data is omnipresent in machine learning. Especially in biomedical segmentation applications, experts need to spend a lot of their time into annotating due to the complexity. Hence, methods to reduce such efforts are desired. *Methods*: Self-Supervised Learning (SSL) is an emerging field that increases performance when unannotated data is present. However, profound studies regarding segmentation tasks and small datasets are still absent. A comprehensive qualitative and quantitative evaluation is conducted, examining SSL's applicability with a focus on biomedical imaging. We consider various metrics and introduce multiple novel application-specific measures. All metrics and state-of-the-art methods are provided in a directly applicable software package. *Results*: We show that SSL can lead to performance improvements of up to 10\%, which is especially notable for methods designed for segmentation tasks. *Conclusion*: SSL is a sensible approach to data-efficient learning, especially for biomedical applications, where generating annotations requires much effort. Additionally, our extensive evaluation pipeline is vital since there are significant differences between the various approaches. *Significance*: We provide biomedical practitioners with an overview of innovative data-efficient solutions and a novel toolbox for their own application of new approaches. Our pipeline for analyzing SSL methods is provided as a ready-to-use software package.


## Project Structure
Overview of this repository:
```
.
├── DLIP
│   ├── data #  Contains the defined datasets as PyTorch Lightning DataModules & Datasets.   
│   ├── experiments #  Contains experiment configurations as yaml files.
│   ├── models #  Contains the defined models as PyTorch Modules.    
│   ├── objectives #  Contains the defined objectives as PyTorch Modules.
│   ├── scripts #  Contains the training and inference scripts.
│   └── utils #  Contains utils functions, which can be used by all modules.
```

The training (`DLIP/scripts/train.py`) and inference script (`DLIP/scripts/inference.py`) are configured by the defined experiments (`DLIP/experiments`) and  utilize the defined datamodules (`DLIP/data`), models (`DLIP/models`) and objectives (`DLIP/objectives`).

## Install
### Prerequisite
- Python == 3.8.5
- Pip == 21.2.4
### Conda Environment
`conda create --name YOUR_ENV_NAME -f snv_ssl.yml`
### Pip Installation
1. Run `pip install -e .`
