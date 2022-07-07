# Weakly Supervised Active Learning
Combination of Weakly Supervised Learning and Active Learning for 3D Point Cloud Classification with Minimum Labeling Effort.

![CoverOld](https://user-images.githubusercontent.com/51992212/177754111-87d2856a-0e3b-4dc9-9c2d-def293ef91be.png)
Credit: Lin, Y., G. Vosselman, and M. Y. Yang (2022). "Weakly supervised semantic segmentation of airborne laser scanning point clouds", Figure 1

## Installation
Please read the `INSTALL.md` file for installation instructions.

## Overview
This repository includes the processing code for training a classification network with weak-region labels and the subsequent segmentation network trained on pseudo labels. The project is structured as follows:

```
 WeaSAL/
    │
    ├── cpp_wrappers/  
    │   ├── cpp_neighbors/...
    │   ├── cpp_subsampling/... 
    │   ├── cpp_utils/... 
    │   └── compile_wrappers.sh 
    ├── datasets/  
    │   ├── common.py
    │   ├── DALES_PseudoLabel.py
    │   ├── DALES_WeakLabel.py
    │   ├── Vaihingen3D_PseudoLabel.py
    │   └── Vaihingen3D_WeakLabel.py
    ├── kernels/  
    │   ├── architectures.py
    │   └── blocks.py
    ├── models/  
    │   ├── dispositions/k_015_center_3D.ply
    │   └── kernel_points.py
    ├── utils/  
    │   ├── anchors.py
    │   ├── ...
    │   └── visualizer.py
    ├── INSTALL.md
    ├── LICENSE
    ├── plot_convergence.py
    ├── pseudoLabel_refinement.py
    ├── README.md
    ├── test_models.py
    ├── train_DALES_PseudoLabel .py
    ├── train_DALES_WeakLabel.py
    ├── train_Vaihingen3D_PseudoLabel.py
    └── train_Vaihingen3D_WeakLabel.py
```

## How to use
*work in progress*


## License
The code is released under MIT License (see LICENSE file for details).
