#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test with weak region labels
#      - adapted by Johannes Ernst
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Confusion matrix function
import utils.conf_matrix as conf_matrix

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix
import pickle

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTesterWL:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False, active_learning=False, test_on_train=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if not active_learning:
            if config.saving:
                test_path = join('test/WeakLabel', config.saving_path.split('/')[-1])
                if not exists(test_path):
                    makedirs(test_path)
                if not exists(join(test_path, 'predictions')):
                    makedirs(join(test_path, 'predictions'))
                if not exists(join(test_path, 'probs')):
                    makedirs(join(test_path, 'probs'))
                if not exists(join(test_path, 'potentials')):
                    makedirs(join(test_path, 'potentials'))
            else:
                test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                self.logits, self.class_logits, self.cam = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(self.logits).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU when reaching number of desired votes
                all_pseudo_lbs = dict()
                all_pseduo_probs = dict()

                if last_min > num_votes:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                        # Get pseudo labels
                        fn = file_path.split('/')[-1].split('.txt')[0]
                        unlabel_pts = self.test_probs[i][:, 0] == 0
                        all_pseduo_probs[fn] = self.test_probs[i]
                        pseudo_lbs = np.argmax(self.test_probs[i], axis=1) + 1
                        pseudo_lbs[unlabel_pts] = 0
                        all_pseudo_lbs[fn] = pseudo_lbs

                    # Save pseudo labels
                    if not active_learning:
                        pl_path = join(test_path, '_pseudo.pickle')
                        prob_path = join(test_path, '_probs.pickle')
                        with open(pl_path, 'wb') as fff:
                            pickle.dump(all_pseudo_lbs, fff)
                        with open(prob_path, 'wb') as fff:
                            pickle.dump(all_pseduo_probs, fff)

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Normal test case 
                    if not active_learning:

                        # Show vote results
                        if test_loader.dataset.set == 'validation':
                            print('Confusion on full clouds')
                            t1 = time.time()
                            Confs = []
                            for i, file_path in enumerate(test_loader.dataset.files):

                                # Insert false columns for ignored labels
                                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                    if label_value in test_loader.dataset.ignored_labels:
                                        proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)
                                    
                                # Get the predicted labels
                                preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                                # Confusion
                                targets = test_loader.dataset.validation_labels[i]
                                Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                            t2 = time.time()
                            print('Done in {:.1f} s\n'.format(t2 - t1))

                            # Regroup confusions
                            C = np.sum(np.stack(Confs), axis=0)

                            # Remove ignored labels from confusions
                            for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                                if label_value in test_loader.dataset.ignored_labels:
                                    C = np.delete(C, l_ind, axis=0)
                                    C = np.delete(C, l_ind, axis=1)

                            IoUs = IoU_from_confusions(C)
                            mIoU = np.mean(IoUs)
                            s = '{:5.2f} | '.format(100 * mIoU)
                            for IoU in IoUs:
                                s += '{:5.2f} '.format(100 * IoU)
                            print('-' * len(s))
                            print(s)
                            print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        Confs = np.zeros((config.num_classes, config.num_classes), dtype=np.int32)
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Get points from file and add coordinate offset
                            points = test_loader.dataset.load_evaluation_points(file_path)
                            points = points + test_loader.dataset.coord_offset

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Create error map
                            targets = test_loader.dataset.validation_labels[i]
                            error_map = preds != targets
                            error_map = error_map.astype('int8')

                            # Save plys
                            cloud_name = file_path.split('/')[-1]
                            test_name = join(test_path, 'predictions', cloud_name)
                            write_ply(test_name,
                                    [points, preds, targets, error_map],
                                    ['x', 'y', 'z', 'preds', 'targets', 'error'])
                            test_name2 = join(test_path, 'probs', cloud_name)
                            prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                        for label in test_loader.dataset.label_values]
                            write_ply(test_name2,
                                    [points, proj_probs[i]],
                                    ['x', 'y', 'z'] + prob_names)

                            # Save potentials
                            pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                            pot_name = join(test_path, 'potentials', cloud_name)
                            pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                            write_ply(pot_name,
                                    [pot_points.astype(np.float32), pots],
                                    ['x', 'y', 'z', 'pots'])

                            # Add up confusions for final confusion matrix
                            labels = test_loader.dataset.validation_labels[i].astype(np.int32)
                            Confs += fast_confusion(labels, preds, test_loader.dataset.label_values).astype(np.int32)

                            # Save ascii preds
                            # ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                            # np.savetxt(ascii_name, preds, fmt='%d')

                        # Save confusion matrix
                        cm_path = join(test_path, 'predictions')
                        if not test_on_train:
                            cm_name = test_loader.dataset.name + '_' + test_loader.dataset.set
                        else:
                            cm_name = test_loader.dataset.name + '_train'
                        conf_matrix.plot(Confs, test_loader.dataset.label_to_names, 
                                        cm_path, file_suffix=cm_name,
                                        abs_vals=False, F1=True, iou=True, show=False)

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Active learning case: Find anchors with high entropy sampling score 
                    # (= insecure predictions) to add weak labels in next iteration
                    else:

                        # Loop over all training files
                        for i, file_path in enumerate(test_loader.dataset.cloud_names):

                            # Get the entropy sampling score for all points in the subsampled cloud
                            # (+1e-12 is added to handle logarithm of probability = 0)
                            entropy_scores = -np.sum(all_pseduo_probs[file_path+'.ply'] * 
                                                     np.log2(all_pseduo_probs[file_path+'.ply']+1e-12), axis=1)

                            # Define path to the anchors file for the subsampled cloud
                            anchors_file = join(test_loader.dataset.tree_path, '{:s}_anchors_{:s}.pkl'.format(
                                file_path, test_loader.dataset.config.anchor_method))

                            # Read pkl from path to get anchors
                            with open(anchors_file, 'rb') as f:
                                anchor, anchor_tree, anchors_dict, anchor_lb = pickle.load(f)

                            # Load the indices of the anchors of the previous iteration
                            anchors_subsampled_file = join(test_loader.dataset.tree_path, '{:s}_subsampled_anchors.pkl'.format(file_path))
                            with open(anchors_subsampled_file, 'rb') as f:
                                anchor_inds_sub = pickle.load(f)

                            # Count class occurrences for all used anchors
                            label_sum = np.zeros([np.size(anchor_lb[0])], dtype=np.int)
                            for label in anchor_inds_sub:
                                label_sum += anchor_lb[label]

                            # Calculate a class score based on occurrences
                            class_scores = np.exp(-label_sum/len(anchor_inds_sub))

                            # Loop over all anchors to get the average entropy sampling score
                            anchor_avg_score = np.zeros(len(anchors_dict)).astype(np.float32)
                            for idx, anchor in enumerate(anchors_dict):

                                # Find the indices (for the subsampled cloud) of the points in the anchor
                                anchor_point_ids = np.squeeze(anchors_dict[anchor][0])

                                # Grab the entropy sampling scores for the points in the anchor
                                anchor_entropy_score = entropy_scores[anchor_point_ids]

                                # Calculate the class weighting score for the anchor based on all occuring classes
                                anchor_class_score = np.matmul(anchor_lb[idx], class_scores)

                                # Calcualte average entropy sampling score multiplied with weighting score for the anchor
                                anchor_avg_score[anchor] = np.mean(anchor_entropy_score)*anchor_class_score

                            # Sort anchors according to their entropy sampling score (descending) and save indices
                            sort_ids = np.argsort(-anchor_avg_score)
                            
                            # Remove already used anchors from the sorted anchor list
                            for used_anchor in anchor_inds_sub:
                                sort_ids = np.delete(sort_ids, np.where(sort_ids == used_anchor))

                            # Select the anchor indices with the highest entropy sampling score
                            if len(sort_ids) > test_loader.dataset.config.added_labels_per_epoch:
                                high_score_ids = sort_ids[0:test_loader.dataset.config.added_labels_per_epoch]
                            else:
                                raise ValueError('Not enough weak labels left for the next iteration')

                            # Add the high entropy sampling score anchor ids to the index list
                            anchor_inds_sub = np.append(anchor_inds_sub, high_score_ids)

                            # Save the updated indices of the subsampled anchors as pickle file
                            with open(anchors_subsampled_file, 'wb') as f:
                                pickle.dump(anchor_inds_sub, f)

            test_epoch += 1

            # Empty cache for more memory
            torch.cuda.empty_cache()

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return
