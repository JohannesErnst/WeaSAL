# ----------------------------------------------------------------------------------------------------------------------
#
#      Refine pseudo labels with ground truth weak region-labels.
#      Output pseudo labels will be used to train the segmentation network.
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

from utils.mayavi_visu import *
from sklearn.neighbors import NearestNeighbors
import glob


# ----------------------------------------------------------------------------------------------------------------------
#
#                   Main Code
#       \******************************/

# Configure folder, model and checkpoint
base_folder = 'test/'
weak_label_log = 'Log_2022-06-04_15-35-23'
data_folder = 'data/Vaihingen3D'
subsampling = 'input_0.240'
sub_folder = join(data_folder,subsampling)

# Define threshold (in percent) for ignoring uncertain labels
threshold = 20
    
# Loop over files in prediction folder
base_path = join(base_folder, weak_label_log)
file_list = glob.glob(base_path+'/predictions/*.ply')
for file in file_list:

    # Read the data from prediction file
    data = read_ply(file)
    points = np.array([data['x'],data['y'],data['z']]).T
    pseudo_lbs = data['preds']
    file_name = file.split('/')[-1].split('.ply')[0]

    # Read the data from the original file
    file_orig = join(sub_folder, file_name + '.ply')
    data_orig = read_ply(file_orig)
    points_orig = np.array([data_orig['x'],data_orig['y'],data_orig['z']]).T
    
    # Get point neighbours and indices of points in original cloud
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points[:,:3])
    distance, indices = nbrs.kneighbors(points_orig[:,:3])    
    indices = np.squeeze(indices)
    
    # Get probabilities of predicted labels
    prob_path = join(base_path, 'probs/' + file_name + '.ply')
    data = read_ply(prob_path)       
    probs = np.vstack((data['Powerline'], data['LowVegetation'], data['ImperviousSurfaces'],  # would be nice to read this in -jer
                        data['Car'], data['Fence/Hedge'], data['Roof'], data['Facade'],
                        data['Shrub'], data['Tree'])).T        
            
    # Refine probabilities with ground truth weak labels of subsampled cloud
    weak_label = np.genfromtxt(join(sub_folder, 'weak_labels_perPoints.txt'), delimiter=' ') 
    probs = probs[indices]
    probs = probs*weak_label
    
    # Select only confident (i.e. with high probability) predictions
    # and assign "no-label" (here: 10) to the rest of the points
    empty = np.max(probs, axis=-1) < (0.01*threshold)
    pseudo_lbs = pseudo_lbs[indices]
    pseudo_lbs[empty] = 10

    # Assign weights based on label occurance
    unique_lbs, counts = np.unique(pseudo_lbs, return_counts=True) 
    counts = counts[:9]
    weights = np.log(1/(counts/np.sum(counts)))
    weights_norm = weights/np.sum(weights)        

    # Save refined pseudo labels and weights for subsampled cloud in separate folder
    pseudo_folder = join(data_folder, 'PseudoLabels')
    if not exists(pseudo_folder):
        makedirs(pseudo_folder)
    out_folder = join(pseudo_folder, weak_label_log)
    if not exists(out_folder):
        makedirs(out_folder)

    pseudo_path = join(out_folder, file_name+'_t'+str(threshold)+'_pseudo.txt')
    np.savetxt(pseudo_path, pseudo_lbs, fmt='%i')

    weights_path = join(out_folder, file_name+'_t'+str(threshold)+'_weight.txt')
    np.savetxt(weights_path, weights_norm, fmt='%.3f')

    print('Created: ' + pseudo_path)
    print('Created: ' + weights_path)
