# ----------------------------------------------------------------------------------------------------------------------
#
#      Refine pseudo labels with ground truth weak region-labels.
#      Output pseudo labels will be used to train the segmentation network.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Johannes Ernst - 2022
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join, exists

# Other scripts
from sklearn.neighbors import NearestNeighbors
from utils.anchors import *

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def get_weak_labels_per_point(cloud_name, sub_folder, sub_radius, num_classes, anchor_method):
    """
    Create point-wise weak labels for the given input cloud. 
    Uses subcloud labels and overlap region labels as weak labels.

    :param cloud_name: Input cloud to create the weak labels for
    :param sub_folder: Folder to subsampled data
    :param sub_radius: Radius of the subclouds
    :param num_classes: Number of classes for dataset
    :param anchor_method: Method for placing the anchors/subclouds
    :return: weak_labels
    """

    # Name of the input files
    KDTree_file = join(sub_folder, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(sub_folder, '{:s}.ply'.format(cloud_name))

    # Get data and read pkl with search tree
    if not exists(KDTree_file):
        raise ValueError('KDTree file does not exist: {:s}'.format(KDTree_file))
    data = read_ply(sub_ply_file)
    sub_labels = data['class']
    with open(KDTree_file, 'rb') as f:
        search_tree = pickle.load(f)

    # Define the same anchors (i.e. subregions of point cloud) as when training/testing
    points = np.array(search_tree.data)
    anchor = get_anchors(points, sub_radius, method=anchor_method)
    anchor, anchor_tree, anchor_dict, anchor_lb = anchors_with_points(
        search_tree, anchor, sub_labels, sub_radius, num_classes)

    # Update subregion information according to overlaps
    anchor, anchor_tree, anchor_dict, anchor_lb = update_anchors(
        search_tree, anchor, anchor_tree, anchor_dict, anchor_lb, sub_radius)

    # Create point-wise weak (subcloud) labels
    weak_labels = np.ones((points.shape[0], num_classes))
    for aa in anchor_dict.keys():
        idx = anchor_dict[aa][0]
        lbs = anchor_lb[aa]
        slc_lb = weak_labels[tuple(idx)]
        weak_labels[tuple(idx)] = slc_lb*lbs

    return weak_labels


# ----------------------------------------------------------------------------------------------------------------------
#
#                   Main Code
#       \******************************/


# Define weak label log for pseudo label refinement (from test/WeakLabel)
weak_label_log = 'Log_2022-07-01_14-17-59'

# Define threshold (in percent) for ignoring uncertain labels
threshold = 20

# Load configuration
config_path = join('results/WeakLabel', weak_label_log)
config = Config()
config.load(config_path)

# Set paths and file list (select all training files)
base_path = join('test/WeakLabel', weak_label_log)
data_folder = join('data', config.dataset)[:-2]
sub_folder = join(data_folder, 'input_{:.3f}'.format(config.first_subsampling_dl))
training_files = join(data_folder, 'Training')
refinement_list = [join(base_path, 'predictions', f)
                   for f in listdir(training_files) if isfile(join(training_files, f))]

# Loop over list of files for refinement
print('Pseudo label refinement for ' + weak_label_log + ' with threshold ' + str(threshold) + '%:\n')
counts = np.zeros(config.num_classes, np.int64)
for file in refinement_list:

    # Read the data from prediction file
    data = read_ply(file)
    points = np.array([data['x'], data['y'], data['z']]).T
    pseudo_lbs = data['preds']
    file_name = file.split('/')[-1].split('.ply')[0]

    # Read the data from the original file
    file_orig = join(sub_folder, file_name + '.ply')
    data_orig = read_ply(file_orig)
    points_orig = np.array([data_orig['x'], data_orig['y'], data_orig['z']]).T

    # Get point neighbours and indices of points in original cloud
    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='kd_tree').fit(points[:, :3])
    distance, indices = nbrs.kneighbors(points_orig[:, :3])
    indices = np.squeeze(indices)

    # Get probabilities of predicted labels
    prob_path = join(base_path, 'probs/' + file_name + '.ply')
    data = read_ply(prob_path)
    label_list = data.dtype.names[3:]
    probs = np.vstack([data[label] for label in label_list]).T

    # Refine probabilities with ground truth weak labels of subsampled cloud
    print('Creating point-wise weak labels for "' + file_name + '":')
    weak_labels = get_weak_labels_per_point(
        file_name, sub_folder, config.sub_radius, config.num_classes, config.anchor_method)
    probs = probs[indices]
    probs = probs*weak_labels

    # Select only confident (i.e. with high probability) predictions
    # and assign "no-label" (here: 10) to the rest of the points
    empty = np.max(probs, axis=-1) < (0.01*threshold)
    pseudo_lbs = pseudo_lbs[indices]
    pseudo_lbs[empty] = 10

    # Count labels in file and save for weighting
    unique_lbs, counter = np.unique(pseudo_lbs, return_counts=True)
    for c in range(len(counts)):
        if c in unique_lbs:
            counts[c] = counts[c] + counter[np.where(unique_lbs == c)]

    # Save refined pseudo labels for subsampled cloud in separate folder
    pseudo_folder = join(data_folder, 'PseudoLabels')
    if not exists(pseudo_folder):
        makedirs(pseudo_folder)
    out_folder = join(pseudo_folder, weak_label_log)
    if not exists(out_folder):
        makedirs(out_folder)

    pseudo_path = join(out_folder, file_name+'_t'+str(threshold)+'_pseudo.txt')
    np.savetxt(pseudo_path, pseudo_lbs, fmt='%i')
    print('Created: ' + pseudo_path)

# Create weights based on label occurance for all training files and save as file
if 0 in counts:
    raise ValueError('Pseudo labels are missing classes! Lower threshold or improve weak label training.')
weights = np.log(1/(counts/np.sum(counts)))
weights_norm = weights/np.sum(weights)
weights_path = join(out_folder, config.dataset[:-2]+'_t'+str(threshold)+'_weight.txt')
np.savetxt(weights_path, weights_norm, fmt='%.3f')
print('Created: ' + weights_path + '\n')
