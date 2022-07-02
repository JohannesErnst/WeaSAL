#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on variable datasets
#      - adapted by Johannes Ernst
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.Vaihingen3D_WeakLabel import *
from datasets.Vaihingen3D_PseudoLabel import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester_WeakLabel import ModelTesterWL
from utils.tester_PseudoLabel import ModelTesterPL
from models.architectures import *


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):   # Make it run with /WeakLabel/ check if it works -jer

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_Vaihingen3DWL', 'last_Vaihingen3DPL', 'last_DALESWL', 'last_DALESPL']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs of either weak or pseudo label training
        if test_dataset[-2:] == 'WL':
            logs = np.sort([os.path.join('results/WeakLabel', f) 
                            for f in os.listdir('results/WeakLabel') if f.startswith('Log')])
        else:
            logs = np.sort([os.path.join('results/PseudoLabel', f) 
                            for f in os.listdir('results/PseudoLabel') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_Vaihingen3DWL', 'last_Vaihingen3DPL', 'last_DALESWL', 'last_DALESPL']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > 'results/XLabel/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'results/WeakLabel/Log_2022-06-28_08-27-45'
    chosen_log = 'last_Vaihingen3DWL'

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10
    config.dropout = 0

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'Vaihingen3DWL':
        test_dataset = Vaihingen3DWLDataset(config, set=set, use_potentials=True)
        test_sampler = Vaihingen3DWLSampler(test_dataset)
        collate_fn = Vaihingen3DWLCollate
    elif config.dataset == 'Vaihingen3DPL':
        test_dataset = Vaihingen3DPLDataset(config, set=set, use_potentials=True)
        test_sampler = Vaihingen3DPLSampler(test_dataset)
        collate_fn = Vaihingen3DPLCollate
    elif config.dataset == 'DALES':
        print("Not implemented")
        # test_dataset = DALESDataset(config, set=set, use_potentials=True)
        # test_sampler = DALESSampler(test_dataset)
        # collate_fn = DALESCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model and tester (must match training network model)
    t1 = time.time()
    if config.model_name == 'KPFCNN_mprm':
        net = KPFCNN_mprm(config, test_dataset.label_values, test_dataset.ignored_labels)
        tester = ModelTesterWL(net, chkp_path=chosen_chkp)
    elif config.model_name == 'KPFCNN_mprm_ele':
        net = KPFCNN_mprm_ele(config, test_dataset.label_values, test_dataset.ignored_labels)
        tester = ModelTesterWL(net, chkp_path=chosen_chkp) 
    elif config.model_name == 'KPFCNN':
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels) 
        tester = ModelTesterPL(net, chkp_path=chosen_chkp)
    else:
        raise ValueError('Unsupported model for testing: ' + config.model_name)

    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Testing
    if config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config, num_votes=10)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)