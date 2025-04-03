# Macaca-Star: Fully Automated and Scalable Pipeline for Macaque Brain Registration
  
# Overview
**Macaca-Star**:This project focuses on the registration of macaque brain images and includes a comprehensive set of preprocessing methods for handling and analyzing brain imaging data. The primary goal is to enable accurate alignment of macaque brain images across different modalities, facilitating the study of neuroanatomy and brain structure. The project implements a combination of deep learning models and traditional image processing techniques to handle a variety of challenges, including the removal of imaging artifacts, alignment of anatomical regions, and the integration of multi-modal data.

The [example](./example) folder in this project contains the test data.

<p align="center">
<img src="https://github.com/user-attachments/assets/e850250e-9390-4c54-a3d7-99e8f61e1812" width="800">

# Macaca-Star pipeline
Macaca-Star is an open-source software project developed using the Python. It significantly simplifies the installation process and lowers the barrier to use. The software has been successfully installed and tested on both Windows, Ubuntu 20.04, and Ubuntu 22.04 systems, ensuring excellent compatibility and stability.

# 3D modality tansfer
<p align="center">
<img src="https://github.com/user-attachments/assets/6b105954-14e3-4061-953d-311b27d08b62" width="800">

We provide a robust 3D modality transfer method for 3D fMOST PI and Blockface images, and the [checkpoints](./checkpoints) has already been uploaded to the project. . This method does not require re-training during usage.
# Installation
```Bash
git clone https://github.com/HNU-BIE/Macaca-Star.git

pip install -r requirements.txt
```
