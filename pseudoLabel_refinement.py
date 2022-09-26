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


def get_weak_labels_per_point(cloud_name, sub_folder, anchor_method, num_classes):
    """
    Get point-wise weak labels for the given input cloud. 
    Uses subcloud labels and overlap region labels as weak labels.

    :param cloud_name: Input cloud to create the weak labels for
    :param sub_folder: Folder to subsampled data
    :param anchor_method: Method for selecting the anchors
    :param num_classes: Number of classes for dataset
    :return: weak_labels
    """

    # Name of the input files
    KDTree_file = join(sub_folder, '{:s}_KDTree.pkl'.format(cloud_name))
    anchors_file = join(sub_folder, '{:s}_anchors_{:s}.pkl'.format(cloud_name, anchor_method))

    # Get data and read pkl with search tree
    if not exists(KDTree_file):
        raise ValueError('KDTree file does not exist: {:s}'.format(KDTree_file))
    with open(KDTree_file, 'rb') as f:
        search_tree = pickle.load(f)
        num_points = search_tree.data.shape[0]

    # Define the same anchors (i.e. subregions of point cloud) as when training/testing
    if not exists(anchors_file):
        raise ValueError('Anchors file does not exist: {:s}'.format(anchors_file))
    with open(anchors_file, 'rb') as f:
        anchor, anchor_tree, anchors_dict, anchor_lb = pickle.load(f)

    # Create point-wise weak (subcloud) labels
    weak_labels = np.ones((num_points, num_classes))
    for aa in anchors_dict.keys():
        idx = anchors_dict[aa][0]
        lbs = anchor_lb[aa]
        slc_lb = weak_labels[tuple(idx)]
        weak_labels[tuple(idx)] = slc_lb*lbs

    return weak_labels


# ----------------------------------------------------------------------------------------------------------------------
#
#                   Main Code
#       \******************************/


# Define weak label log for pseudo label refinement (from test/WeakLabel)
weak_label_log = 'Log_2022-09-25_15-22-56'

# Define threshold (in percent) for ignoring uncertain labels.
# NOTE: Values are experimental. Might need adaptation for other datasets.
# Default: 10 for DALES and 20 for Vaihingen3D
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
print('\nPseudo label refinement for ' + weak_label_log + ' with threshold ' + str(threshold) + '%:\n')
counts = np.zeros(config.num_classes, np.int64)
for file in refinement_list:

    # Read the data from prediction file
    data = read_ply(file)
    points = np.array([data['x'], data['y'], data['z']]).T
    pseudo_lbs = data['preds']
    file_name = file.split('/')[-1].split('.ply')[0]

    # Reduce coordinates for numeric stability and convert to float32
    points = (points - np.min(points[:,:], 0)).astype(np.float32)

    # Read the data from the original file
    file_orig = join(sub_folder, file_name + '.ply')
    data_orig = read_ply(file_orig)
    points_orig = np.array([data_orig['x'], data_orig['y'], data_orig['z']]).T

    # Reduce original coordinates to match the offset with the prediction cloud
    points_orig = (points_orig - np.min(points_orig[:,:], 0)).astype(np.float32)

    # Get point neighbours and indices of points in original cloud
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points[:, :3])
    distance, indices = nbrs.kneighbors(points_orig[:, :3])
    indices = np.squeeze(indices)

    # Get probabilities of predicted labels
    prob_path = join(base_path, 'probs/' + file_name + '.ply')
    data = read_ply(prob_path)
    label_list = data.dtype.names[3:]
    probs = np.vstack([data[label] for label in label_list]).T

    # Refine probabilities with ground truth weak labels of subsampled cloud
    print('Getting point-wise weak labels for "' + file_name + '"')
    weak_labels = get_weak_labels_per_point(file_name, sub_folder, config.anchor_method, config.num_classes)
    probs = probs[indices]
    probs = probs*weak_labels

    # Select only confident (i.e. with high probability) predictions
    # and assign "no-label" (here: 10) to the rest of the points
    # NOTE: In the following code (pseudo label training), the use of
    # "10" as "no-label" is hard-coded! Changes might lead to errors
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
    print('\nWARNING:\nPseudo labels are missing classes! Lower threshold or improve weak label training.')
weights = np.log(1/((counts+1)/np.sum(counts)))
weights_norm = weights/np.sum(weights)
weights_path = join(out_folder, config.dataset[:-2]+'_t'+str(threshold)+'_weight.txt')
np.savetxt(weights_path, weights_norm, fmt='%.3f')
print('\nCreated: ' + weights_path + '\n')
