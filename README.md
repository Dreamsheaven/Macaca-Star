# Macaca-Star: Fully Automated and Scalable Pipeline for Macaque Brain Registration

# Contents
&#x2022; [Overview](#Overview)  
&#x2022; [Macaca-Star pipeline](#Macaca-Star-pipeline)  
&#x2022; [3D modality tansfer](#3D-modality-tansfer)  
&#x2022; [System requirements](#System-requirements)   
&#x2022; [Installation](#Installation)  
&#x2022; [License](#License)  

# Overview
**Macaca-Star** is an open-source software project developed entirely in Python, focusing on the registration of macaque brain images. It simplifies the installation process and lowers the barrier to entry. The primary goal is to enable accurate alignment of macaque brain images across different modalities, facilitating the study of neuroanatomy and brain structure. The project includes a comprehensive set of preprocessing methods for handling and analyzing brain imaging data and supports multiple data types, such as fMOST PI images, blockface-based fluorescence sections, and MRI scans. By integrating deep learning models with traditional image processing techniques, the project addresses various challenges, including the removal of imaging artifacts, alignment of anatomical regions, and the integration of multi-modal data for accurate cross-modality registration.

The [example](./example) folder in this project contains the test data.

<p align="center">
<img src="https://github.com/user-attachments/assets/e850250e-9390-4c54-a3d7-99e8f61e1812" width="800">

# 3D modality tansfer

We provide a robust 3D modality transfer method for 3D fMOST PI and Blockface images, and the [checkpoints](./checkpoints) has already been uploaded to the project. . This method does not require re-training during usage.

<p align="center">
<img src="https://github.com/user-attachments/assets/6b105954-14e3-4061-953d-311b27d08b62" width="800">

# System requirements
The software has been successfully installed and tested on Windows, Ubuntu 20.04, and Ubuntu 22.04 systems, ensuring excellent compatibility and stability.  

To ensure smooth operation of the program, a minimum of 32GB of RAM is required. This software includes deep learning models, so it is necessary to have the required environments, such as CUDA, installed for deep learning tasks. If you wish to retrain the 3D CycleGAN model, a high-performance GPU such as the A6000 (48GB) or better is recommended. We also provide pretrained results, which can be found in the "[checkpoints](./checkpoints)" folder.

Minimum 32GB RAM required for optimal performance

For model training (3D CycleGAN):
Recommended: NVIDIA RTX A6000 (48GB) or better

# Installation
```Bash
git clone https://github.com/HNU-BIE/Macaca-Star.git

pip install -r requirements.txt
or
conda env create -f enviroment.yml
```
# Getting Started
### Download Checkpoints
Download the model [checkpoints](./checkpoints). All model checkpoints are saved in the "checkpoints" folder within the program directory.
# License
This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

