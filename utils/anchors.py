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
from utils.mayavi_visu import *


def get_anchors(points, sub_radius, xyz_offset=[0,0,0], method='full'):
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
        x_num = np.linspace(x_min, x_max, x_step.astype('int'))+xyz_offset[0]
        y_num = np.linspace(y_min, y_max, y_step.astype('int'))+xyz_offset[1]
        z_num = np.linspace(z_min, z_max, z_step.astype('int'))+xyz_offset[2]
        for x in x_num:
            for y in y_num:
                for z in z_num:
                    n_anchors.append([x, y, z])

    elif method == 'reduced':
        # This method uses half of the anchors used in 'full'
        x_step = np.floor((x_max - x_min) / (2*sub_radius)) + 1
        y_step = np.floor((y_max - y_min) / (2*sub_radius)) + 1
        z_step = np.floor((z_max - z_min) / (2*sub_radius)) + 1  
        x_num = np.linspace(x_min, x_max, x_step.astype('int'))+xyz_offset[0]
        y_num = np.linspace(y_min, y_max, y_step.astype('int'))+xyz_offset[1]
        z_num = np.linspace(z_min, z_max, z_step.astype('int'))+xyz_offset[2]
        for x in x_num:
            for y in y_num:
                for z in z_num:
                    n_anchors.append([x, y, z])
                    n_anchors.append([x, y, z + sub_radius])
                    n_anchors.append([x + sub_radius, y + sub_radius, z])
                    n_anchors.append([x + sub_radius, y + sub_radius, z + sub_radius])
                 
                    
    return np.array(n_anchors)

def anchors_with_points(input_tree, anchors, lbs, radius, n_class):
    # This was once named anchors_part_lbs. Just fyi when trying to find the function -jer
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
            anchor_lbs[cc] = cloud_labels    
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
                anchor_lbs[cc] = anchor_lbs[idx] * anchor_lbs[nei]
                clean_anchors = np.vstack((clean_anchors, np.expand_dims(new_anchor, axis=0)))
                cc = cc+1
                
    print('Anchors considering overlaps: {:.0f}\n'.format(cc))
    anchor_tree = KDTree(clean_anchors, leaf_size=10)
    return clean_anchors, anchor_tree, anchors_dict, anchor_lbs