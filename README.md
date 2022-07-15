# Weakly Supervised Active Learning
Combination of Weakly Supervised Learning and Active Learning for 3D Point Cloud Classification with Minimum Labeling Effort.

*REPO IS WORK IN PROGRESS*

![Overview](https://user-images.githubusercontent.com/51992212/178697796-4d9249c8-599e-4803-9ec8-cb0ae0f163e2.png)

Credit: Lin, Y., G. Vosselman, and M. Y. Yang (2022). "Weakly supervised semantic segmentation of airborne laser scanning point clouds", Figure 1


## Installation
Please read the `INSTALL.md` file for installation instructions.


## Overview
This repository includes the processing code for training a classification network with weak-region labels and the subsequent segmentation network trained on pseudo labels. 

As a Convolutional Neural Network, Kernel Point Convolution ([KPConv](https://arxiv.org/abs/1904.08889)) is used:
https://github.com/HuguesTHOMAS/KPConv-PyTorch

The weak supervision approach follows Multi-Path Region Mining ([MPRM](https://arxiv.org/abs/2003.13035)):
https://github.com/plusmultiply/mprm

Main structure for refined Weakly Supervised Semantic Segmentation ([Weak_ALS](https://www.sciencedirect.com/science/article/pii/S0924271622000661)) is based on:
https://github.com/yaping222/Weak_ALS


## How to use
Create a ```data/``` folder in ```WeaSAL/``` directory that holds the datasets (in this case Vaihingen3D and DALES) for the project:
```
 WeaSAL/
    │
    ├── cpp_wrappers/...  
    ├── data/
    │   ├── DALES/...
    │   │   ├── 5080_54435.ply  
    │   │   ...  
    │   │   └── test_5175_54395.ply
    │   └── Vaihingen3D/
    │       ├── Vaihingen3D_Testing.ply 
    │       └── Vaihingen3D_Training.ply
    ├── datasets/...  
    .
    .
    .
```
- Training the network on weak subcloud labels (```train_Vaihingen3D_WeakLabel.py``` or ```train_DALES_WeakLabel.py```)
- Visualizing convergence of the training (```plot_convergence.py```)
- Generating point-wise pseudo labels (```test_models.py``` with weak-supervision network input and training files)
- Performing pseudo label refinement (```pseudoLabel_refinement.py```)
- Training the network on pseudo labels (```train_Vaihingen3D_PseudoLabel.py``` or ```train_DALES_PseudoLabel.py```)
- Testing the network on the test set (```test_models.py``` with pseudo-supervision network input and test files)


## License
The code is released under MIT License (see LICENSE file for details).
