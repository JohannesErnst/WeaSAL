# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to calculate confusion matrices for KPConv validation output
#      - Johannes Ernst, 2022
#
# ----------------------------------------------------------------------------------------------------------------------

# Import libs and files
import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import numpy.matlib
import utils.conf_matrix as conf_matrix
from utils.ply import read_ply

# Define dataset
dataset = 'DALES'
dataset = 'Vaihingen3D'

# Dialog to open the validation.ply file
root = tk.Tk()
root.withdraw()
filename = filedialog.askopenfilename(title="Select cloud with predictions and ground truth (.ply)", initialdir="/home/valentin/WeaSAL/results")

# Read the data, define output parameters and classes and save output
data = read_ply(filename)
splitString = filename.split("/")
out_folder = "/".join(splitString[:-1])

if dataset == 'Vaihingen3D':
    name = "Vaihingen3D_Val"
    classes = {0: 'Powerline',
               1: 'LowVegetation',
               2: 'ImperviousSurfaces',
               3: 'Car',
               4: 'Fence/Hedge',
               5: 'Roof',
               6: 'Facade',
               7: 'Shrub',
               8: 'Tree'}
    cm = conf_matrix.create(data['class'], data['preds'], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8])
    conf_matrix.plot(
            cm, classes, out_folder, file_suffix=name,
            abs_vals=False, F1=True, iou=True, show=False)

elif dataset == 'DALES':
    name = "DALES_Val"
    classes = {0: 'Unknown',
               1: 'Ground',
               2: 'Vegetation',
               3: 'Cars',
               4: 'Trucks',
               5: 'Power',
               6: 'Poles',
               7: 'Fences',
               8: 'Buildings'}
    cm = conf_matrix.create(data['class'], data['preds'], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8])
    conf_matrix.plot(
            cm, classes, out_folder, file_suffix=name,
            abs_vals=False, F1=True, iou=True, show=False, ignore_labels=[0])
else:
    raise ValueError('Unsupported dataset : ' + dataset)
    