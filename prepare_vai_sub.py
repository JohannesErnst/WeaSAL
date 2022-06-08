# ----------------------------------------------------------------------------------------------------------------------
#
#      Create weak (subcloud) labels and overlap region labels with input clouds and
#      assign weak labels to all points
#      - adapted by Johannes Ernst
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yaping LIN - 2021
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import numpy as np
import pickle

# OS functions
from os.path import exists, join

# Scripts
# from utils.mayavi_visu import *
from utils.anchors import *

# Maybe this file should be included in train or test at some point or combined with vai_pseudo.py -jer
# Mayve we can also read in the config file like done in test_models.py -jer

# ----------------------------------------------------------------------------------------------------------------------
#
#                   Main Code
#       \******************************/

# Define cloud names and path to tree files
cloud_names =  ['Vaihingen3D_Training']            # right now it looks like this can be used with multplie clouds but I don't think so, look at "assign weak label" which should be inside a cloud for loop as well
tree_path = 'data/Vaihingen3D/input_0.240'      # path selection needs to be refined -jer

input_trees = []
input_labels = []
all_data = dict()

# Loop over all clouds
for i, f in enumerate(cloud_names):
    cloud_name = cloud_names[i]

    # Name of the input files
    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

    # Get data and read pkl with search tree
    if not exists(KDTree_file):
        raise ValueError('KDTree file does not exist: {:s}'.format(KDTree_file))
    data = read_ply(sub_ply_file)
    sub_labels = data['class']        
    with open(KDTree_file, 'rb') as f:
        search_tree = pickle.load(f)

    # Fill data containers
    input_trees += [search_tree]
    input_labels += [sub_labels]

# Define the same anchors (i.e. subregions of point cloud) as when training/testing
sub_radius = 6              # this should be set automatically according to train files -jer
num_classes = 9
anchor_method = 'reduce1'
for i, tree in enumerate(input_trees):
    points = np.array(tree.data)
    anchor = get_anchors(points, sub_radius, method=anchor_method)
    anchor, anchor_tree, anchors_dict, achor_lb = anchors_with_points(
        tree, anchor, input_labels[i], sub_radius, num_classes)

    # Update subregion information according to overlaps
    lbs = input_labels[i]
    anchor, anchor_tree, anchors_dict, achor_lb = update_anchors(
        tree, anchor, anchor_tree, anchors_dict, achor_lb, sub_radius)

    # Save anchors              # do we even need this? Variable is not used anymore. MAybe if we use multiple clouds the next part with all_class_lbs wont work... -jer
    # c_name = cloud_names[i]
    # all_data[c_name] = anchor, anchor_tree, anchors_dict, achor_lb
    
# Assign weak (subcloud) labels to all points
cloud_labels_all = np.ones((points.shape[0], num_classes))
for aa in anchors_dict.keys():
    idx = anchors_dict[aa][0]
    lbs = achor_lb[aa]
    slc_lb = cloud_labels_all[tuple(idx)]
    cloud_labels_all[tuple(idx)] = slc_lb*lbs
    
# Save weak labels (for each point) as file
class_lb = join(tree_path, 'weak_labels_perPoints.txt')
np.savetxt(class_lb, cloud_labels_all, fmt='%i', delimiter=' ')
