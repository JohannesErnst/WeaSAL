#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script for processing anchors (i.e. origin points of spherical subclouds) 
#      for weak region-labels
#      - adapted by Johannes Ernst
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yaping LIN - 03/08/2021
#
# ----------------------------------------------------------------------------------------------------------------------

# Import libs
import numpy as np
import random
from utils.mayavi_visu import *


def get_anchors(points, sub_radius, method='full'):
    """
    Return anchor points (n_anchors) for specified pointcloud
    """
    n_anchors = []

    # Define boundaries of pointcloud
    x_max = points[:, 0].max()
    x_min = points[:, 0].min()
    y_max = points[:, 1].max()
    y_min = points[:, 1].min()
    z_max = points[:, 2].max()
    z_min = points[:, 2].min()

    # Find regularly spaced anchor positions 
    if method == 'full':
        # This method uses anchors with a spacing of sub_radius
        x_step = np.floor((x_max - x_min) / sub_radius) + 1
        y_step = np.floor((y_max - y_min) / sub_radius) + 1
        z_step = np.floor((z_max - z_min) / sub_radius) + 1  
        x_num = np.linspace(x_min, x_max, x_step.astype('int'))
        y_num = np.linspace(y_min, y_max, y_step.astype('int'))
        z_num = np.linspace(z_min, z_max, z_step.astype('int'))
        for x in x_num:
            for y in y_num:
                for z in z_num:
                    n_anchors.append([x, y, z])

    elif method == 'reduced':
        # This method uses half of the anchors used in 'full'
        x_step = np.floor((x_max - x_min) / (2*sub_radius)) + 1
        y_step = np.floor((y_max - y_min) / (2*sub_radius)) + 1
        z_step = np.floor((z_max - z_min) / (2*sub_radius)) + 1  
        x_num = np.linspace(x_min, x_max, x_step.astype('int'))
        y_num = np.linspace(y_min, y_max, y_step.astype('int'))
        z_num = np.linspace(z_min, z_max, z_step.astype('int'))
        for x in x_num:
            for y in y_num:
                for z in z_num:
                    n_anchors.append([x, y, z])
                    n_anchors.append([x, y, z + sub_radius])
                    n_anchors.append([x + sub_radius, y + sub_radius, z])
                    n_anchors.append([x + sub_radius, y + sub_radius, z + sub_radius])     

    else:
        raise ValueError('Unsupported method (' + method + ') for creating anchor points')
                    
    return np.array(n_anchors)

def anchors_with_points(input_tree, anchors, lbs, radius, n_class):
    """
    Return anchors (i.e. subregions) that have points inside
    """
    clean_anchors = []
    anchors_dict = dict()
    anchor_lbs = dict()
    cc = 0
    for i in range(anchors.shape[0]):

        # Collect number of points in subregion
        center_point = anchors[i].reshape(1, -1)
        input_inds = input_tree.query_radius(center_point, r=radius)[0]
        n = input_inds.shape[0]

        # Save anchors with points 
        if n > 0 :
            clean_anchors += [anchors[i]]
            anchors_dict[cc] = [[input_inds], [anchors[i]]]
            slc_lbs = lbs[input_inds]
            cls_lbs = np.unique(slc_lbs)
            cloud_labels = np.zeros((n_class))
            cloud_labels[cls_lbs] = 1
            anchor_lbs[cc] = cloud_labels.astype(int)
            cc = cc + 1
            
    clean_anchors = np.array(clean_anchors)
    anchor_tree = KDTree(clean_anchors, leaf_size=10)
    return clean_anchors, anchor_tree, anchors_dict, anchor_lbs

def update_anchors(input_tree, clean_anchors, anchor_tree, anchors_dict, anchor_lbs, sub_radius):
    """
    Update anchors (i.e. subregions) and labels according to overlap
    """
    cc = len(anchors_dict.keys())
    points = np.array(input_tree.data)
    print('Anchors without considering overlap: {:.0f}'.format(cc))

    # Search neighbouring anchors
    anchor_nei_idx, dists = anchor_tree.query_radius(clean_anchors,
                                                     r = 1.5*sub_radius,
                                                     return_distance = True)    
                                            
    # For all anchors
    for idx in range(len(anchor_nei_idx)):
        nei_mask = anchor_nei_idx[idx]>idx
        neis = anchor_nei_idx[idx][nei_mask]
        i_idxs = anchors_dict[idx][0][0]
        
        # For all neighbours
        for nei in neis:        
            nei_idxs = anchors_dict[nei][0][0]
            overlap = np.in1d(i_idxs, nei_idxs)
            if overlap.sum()<1:
                continue              
            new_idxs = i_idxs[overlap]
    
            # If two subregions have different class labels 
            # store new anchors and its label
            if (anchor_lbs[idx] != anchor_lbs[nei]).sum() > 0:
                new_anchor = np.mean(points[new_idxs], axis=0)
                anchors_dict[cc] = [[new_idxs], [new_anchor]]
                anchor_lbs[cc] = (anchor_lbs[idx] * anchor_lbs[nei]).astype(int)
                clean_anchors = np.vstack((clean_anchors, np.expand_dims(new_anchor, axis=0)))
                cc = cc+1
                
    print('Anchors considering overlaps: {:.0f}\n'.format(cc))
    anchor_tree = KDTree(clean_anchors, leaf_size=10)
    return clean_anchors, anchor_tree, anchors_dict, anchor_lbs

def select_anchors(anchor, anchors_dict, anchor_lb, anchor_inds_sub):
    """
    Function selects only the anchors (i.e. subregions) that are given as 
    indices in anchor_inds_sub. The indices are based on the full anchor set.
    """

    # Use indices to reduce the anchor variables
    anchor_sub = anchor[anchor_inds_sub]
    anchors_dict_sub = dict()
    anchor_lb_sub = dict()
    for idx, anchor_ind in enumerate(anchor_inds_sub):
        anchors_dict_sub[idx] = anchors_dict[anchor_ind]
        anchor_lb_sub[idx] = anchor_lb[anchor_ind]
    anchor_tree_sub = KDTree(anchor_sub, leaf_size=10)

    return anchor_sub, anchor_tree_sub, anchors_dict_sub, anchor_lb_sub

def subsample_anchors(anchor, anchors_dict, anchor_lb, anchor_count, subsample_method):
    """
    Function subsamples anchors (i.e. subregions) for active learning to a given
    amount and returns the updated anchor variables as well as a list with the 
    indices of the subsampled anchors. The indices are based on the full anchor set.

    Subsample_method defines the way we subsample the weak labels:
        - linear:   Produces linearly subsampled anchors
        - random:   Produces randomly selected anchors
        - balanced: Produces equal amount of weak labels for each class
    """

    # Switch between subsample methods
    if subsample_method == 'linear':

        # Linearly subsample the available anchors and save the remaining indices
        anchor_inds_sub = np.round(np.linspace(0, anchor.shape[0]-1, anchor_count)).astype(int)
    
    elif subsample_method == 'random':

        # Create a variable that holds all anchor indices
        anchor_inds = list(range(len(anchor_lb)))

        # Select random anchor indices
        anchor_inds_sub = sorted(random.choices(anchor_inds, k=anchor_count))

    elif subsample_method == 'balanced':

        # Create a dictionary that holds the weak label occurances of all classes
        label_class_counts = dict()

        # Create a variable that holds all anchor indices
        anchor_inds = list(range(len(anchor_lb)))

        # Initialize the dictionary with empty lists to be filled
        for label in range(len(anchor_lb[0])):
            label_class_counts[label] = []

        # Loop over all weak labels to save the occurance of classes
        for key in anchor_lb:
            weak_label = anchor_lb[key]
            classes = np.where(weak_label == 1)[0]
            for idx in classes:
                label_class_counts[idx].append(key)

        # Define approximate number of labels per class (rounded down)
        labels_per_class = int(anchor_count/len(label_class_counts))

        # Pick same amount of labels from each class (if available)
        anchor_inds_sub = []
        for label in label_class_counts:
            if len(label_class_counts[label]) >= labels_per_class:
                per_class_ids = np.round(np.linspace(0, len(label_class_counts[label])-1, labels_per_class)).astype(int)
                anchor_inds_sub += [label_class_counts[label][i] for i in per_class_ids]
            else:
                anchor_inds_sub += label_class_counts[label]

        # Filter label_class_counts to only contain unique labels
        anchor_inds_sub = list(set(anchor_inds_sub))

        # Select remaining anchor indices that are not picked yet
        for sub_ind in anchor_inds_sub:
            anchor_inds.remove(sub_ind)

        # Randomly add remaining amount of labels to reach anchor_count
        remaining_labels = anchor_count-len(anchor_inds_sub)
        random_labels = random.choices(anchor_inds, k=remaining_labels)
        anchor_inds_sub += random_labels

        # Sort the selected labels
        anchor_inds_sub = sorted(anchor_inds_sub)
    
    else:
        raise ValueError('Subsample method "' + subsample_method + '" is not supported!')

    # Use indices to reduce the anchor variables
    anchor_sub, anchor_tree_sub, anchors_dict_sub, anchor_lb_sub = select_anchors(
        anchor, anchors_dict, anchor_lb, anchor_inds_sub)

    return anchor_sub, anchor_tree_sub, anchors_dict_sub, anchor_lb_sub, anchor_inds_sub