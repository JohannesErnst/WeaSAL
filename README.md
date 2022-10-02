# Weakly Supervised Active Learning
Combination of weakly supervised learning and active learning for 3D point cloud classification with minimum labeling effort.

![Overview](https://user-images.githubusercontent.com/51992212/193449535-a0d5eb9f-ffbf-4253-ac92-f46b26529b68.png)

Graphical overview of the WeaSAL network workflow (adapted from Lin, Y., G. Vosselman, and M. Y. Yang (2022): "Weakly supervised semantic segmentation of airborne laser scanning point clouds", Figure 1)


## Installation
Please read the `INSTALL.md` file for installation instructions.


## Overview
This repository includes the processing code for training a classification network with weak-region labels and the subsequent segmentation network trained on pseudo-labels in [PyTorch](https://pytorch.org/).

As a Convolutional Neural Network, Kernel Point Convolution ([KPConv](https://arxiv.org/abs/1904.08889)) is used:<br/>
https://github.com/HuguesTHOMAS/KPConv-PyTorch

The weak supervision approach follows Multi-Path Region Mining ([MPRM](https://arxiv.org/abs/2003.13035)):<br/>
https://github.com/plusmultiply/mprm

Main structure for refined Weakly Supervised Semantic Segmentation ([Weak_ALS](https://www.sciencedirect.com/science/article/pii/S0924271622000661)) is based on:<br/>
https://github.com/yaping222/Weak_ALS

Scripts for training and testing on two datasets are provided, namely *Vaihingen3D* and *DALES*.


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
Workflow in short:<br/>
`train_dataset_WeakLabel.py` &rarr; `test_models.py` &rarr; `pseudoLabel_refinement.py` &rarr; `train_dataset_PseudoLabel.py` &rarr; `test_models.py`

Workflow in full:
- Training the network on weak subcloud labels (```train_Vaihingen3D_WeakLabel.py``` or ```train_DALES_WeakLabel.py```). This will create a "results" folder inside the WeaSAL directory where all training results are stored. 
- Visualizing convergence of the training (```plot_convergence.py```) to plot runtime, overall accuracy on the validation set and loss. Output can be set up inside the script. The script can be used for weak and pseudo-label training results.
- Generating point-wise pseudo-labels (```test_models.py``` with weak-supervision network and training files as input). This will create a "test" folder inside the WeaSAL directory where all test results are stored. Parameters have to be set in accordance to the training log file and the input files. Use:<br/>
  - `chosen_log = 'Log_####'` to your specific weak-supervision log created during the training or use the `chosen_log = 'last_####'` option to automatically retreive the last created log.<br/>
  - `set = 'train'` to create pseudo-labels for the training set.<br/>
- Performing pseudo-label refinement (```pseudoLabel_refinement.py```) to improve the input for the following segmentation network. Creates new pseudo-label and class weighting file (as .txt) inside the corresponding data folder. Parameters have to be set in accordance to the log file and the dataset.
- Training the network on the refined pseudo-labels (```train_Vaihingen3D_PseudoLabel.py``` or ```train_DALES_PseudoLabel.py```). Output is stored inside the "results" folder. 
- Testing the network on the test set (```test_models.py``` with pseudo-supervision network and test files as input). Final testing output is stored inside the "test" folder. Parameters have to be set in accordance to the training log file and the input files. Use:<br/>
  - `chosen_log = 'Log_####'` to your specific pseudo-supervision log created during the training or use the `chosen_log = 'last_####'` option to automatically retreive the last created log.<br/>
  - `set = 'test'` to test the network on the testing set.<br/>

Elaborate comments can be found inside the code.


## License
The code is released under MIT License (see LICENSE file for details).
