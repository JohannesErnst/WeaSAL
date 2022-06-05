#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#      - adapted by Johannes Ernst
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
import numpy as np
from torch_scatter import scatter
from torch.nn import Dropout


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        self.dropout = config.dropout
        if config.dropout:
            self.droplayer = Dropout(p = float(config.dropout))

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        if self.dropout:
            sa = self.droplayer(x)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

    def contrast_loss_w(self, outputs, labels, config, threshold=0.4, gt=None):
            """
            Runs contrastive loss on outputs of the model
            :param outputs: logits predicted by the network
            :param labels: labels
            :return: loss
            """

            # Define parameter
            temperature = 0.1
            base_temperature = 1
            self.pts_loss = 0
            self.pts_loss_self = 0
            slc_con = 1000
            N = outputs.shape[0]        
            tensor_0 = torch.tensor(0).float().cuda()
            threshold = config.contrast_thd 
            
            # Get probabilities
            prob = torch.nn.Softmax(1)(outputs) 
            pseudo_logits = prob.max(1)[0]

            # Gather all labeled points and confident unlabeled points
            label_id = labels < 10
            certain_label = pseudo_logits > threshold
            certain_label = (certain_label + label_id) > 0 

            # Get pseudo labels
            pseudo_lbs = torch.argmax(prob, dim=1)
            pseudo_lbs[label_id] = labels[label_id] 
            all_valid_idx = torch.where(certain_label)[0]

            # Random slc 2000 indx
            nn = all_valid_idx.shape[0] 
            if nn < 1:
                print('skipped')
                return tensor_0
            if nn >= slc_con:
                slc_idx_idx = torch.randint(0,nn,(slc_con,))
                slc_idx = all_valid_idx[slc_idx_idx]
            else:
                o_idx = torch.arange(nn)
                slc_idx_idx =  torch.randint(0,nn,(slc_con-nn,))
                slc_idx_idx = torch.stack((o_idx, slc_idx_idx), dim = 0)
                slc_idx = all_valid_idx[slc_idx_idx]

            # Create a mask
            mask1 = torch.ones(N, slc_con)
            all_idx = torch.arange(N).cuda()
            idx = torch.where(all_idx[:, None] == slc_idx[None, :])
            cc = torch.stack(idx, dim=0).T
            cc = cc.cuda()
            mask1[cc[:,0], cc[:,1]] = 0
            mask1 = mask1.cuda()
                    
            pseudo_label_slc = pseudo_lbs[slc_idx]
            
            certain_label_slc = certain_label[slc_idx] 
            
            mask_certaion = certain_label_slc.unsqueeze(0) == certain_label.unsqueeze(-1) 
            
            pos_mask = pseudo_label_slc.unsqueeze(0) == pseudo_lbs.unsqueeze(-1) * mask1 * mask_certaion
            # outputs = F.normalize(outputs, dim=1)               # what is F? Maybe this should be nn.functionals.normalize -jer
            outputs = nn.functionals.normalize.normalize(outputs, dim=1)    
            exit("Stop, once the code reaches this part double check the F.normalize part. Does it work (in Stefans code)? -jer")
            x_slc = outputs[slc_idx]
            
            mul = torch.div(
                torch.matmul(outputs, x_slc.T),
                temperature)
            eps = 1e-8

            # For numerical stability
            logits_max, _ = torch.max(mul, dim=1, keepdim=True)
            logits = mul - logits_max.detach()

            # Compute logarithmic probability
            exp_logits = torch.exp(logits) * (mask1 * mask_certaion) # (0,1]
            log_prob = (logits - torch.log((exp_logits.sum(1, keepdim=True)+eps)))* (mask1 * mask_certaion) 

            # Compute mean of log-likelihood over positive samples
            # If no positve samples are found, take them as 0
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1)+1e-12)
            self.pts_loss = - (temperature / base_temperature) * mean_log_prob_pos
            cal_slc = self.pts_loss>0
            self.pts_loss = self.pts_loss[cal_slc]
            pseudo_lbs = pseudo_lbs[cal_slc]
            self.pts_loss = scatter(self.pts_loss, pseudo_lbs, reduce="mean")
            self.pts_loss = self.pts_loss[self.pts_loss>0]

            return self.pts_loss.mean()


class KPFCNN_mprm(nn.Module):
    """
    Class defining KPFCNN for weak labels with multi-path region mining (mprm)
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN_mprm, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.features_layer = 'encoder_blocks.5.unary_shortcut.mlp'
        self.forward_features = {}
        self.backward_features = {}

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global', 'attention']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'attention' in block:
                break            
            if 'upsample' in block:
                break
                      
            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config)) # r is radius

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        
        self.multi_att = dual_att('attention', out_dim, out_dim, r, layer, config)
        self.no_ga = global_average_block('ga1', config.num_classes, config.num_classes, layer, config)
        self.da_ga = global_average_block('ga2', config.num_classes, config.num_classes, layer, config)
        self.spa_ga = global_average_block('ga3', config.num_classes, config.num_classes, layer, config)
        self.cha_ga = global_average_block('ga4', config.num_classes, config.num_classes, layer, config)
        

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss          # clear this part up once it is running. Might try to change this for better results though (look at original KPConv) -jer
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            self.criterion_multi = torch.nn.BCEWithLogitsLoss(weight=class_w)
        else:           
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
            self.criterion_multi = torch.nn.BCEWithLogitsLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        self.register_hooks()

        return

    def register_hooks(self):
        """ Register forward and backward hooks that store features and gradients from the given layer """
        def forward_hook(_module, _forward_input, forward_output):
            self.forward_features[forward_output.device] = forward_output
            
        def backward_hook(_module, _backward_input, backward_output):
            self.backward_features[backward_output[0].device] = backward_output[0]

        for name, module in self.named_modules():
            if name == self.features_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive encoder blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
            
        # Combine attention modules and save logits
        spa_att, cha_att, no_att, dual_att = self.multi_att(x, batch)

        no_att_cla = self.no_ga(no_att, batch)
        dual_att_cla = self.da_ga(dual_att, batch)
        spa_att_cla = self.spa_ga(spa_att, batch)
        cha_att_cla = self.cha_ga(cha_att, batch)  
        
        cla_logits = [no_att_cla, dual_att_cla, spa_att_cla, cha_att_cla]
        
        # Loop over consecutive decoder blocks
        for block_i, block_op in enumerate(self.decoder_blocks):
            no_att = block_op(no_att, batch)
            dual_att = block_op(dual_att, batch)
            spa_att = block_op(spa_att, batch)
            cha_att = block_op(cha_att, batch)

        # Element-wise maximum to get pseudo labels
        x = torch.max(no_att, dual_att)
        x = torch.max(x, spa_att)
        x = torch.max(x, cha_att)
            
        cam = [no_att, dual_att, spa_att, cha_att]
        
        return x, cla_logits, cam
   
    def class_logits_loss(self, class_logits, cloud_lb):
        # I think this loss can be deleted at some point because it was used by Wei et al
        # and doesn't consider the overlap region loss. If I delete this, I may completely remove
        # the variables cloud_lb and cloud_all_lb from Vaihingen3D_WeakLabel because they are only
        # weak labels used for the whole input sphere (in_radius) and not the smaller subclouds 
        # (i.e. subradius). Then only the variables region_lb and region will be left. This should
        # be a little less confusing in Vaihingen3d_WeakLabel.py. -jer
        """
        Runs the BCEWithLogitsLoss (binary cross entropy) on outputs of the model
        :param class_logits: logits
        :param cloud_lb: labels
        :return: loss
        """

        # BCEWithLogitsLoss on class logits
        self.output_loss1 = self.criterion_multi(class_logits[0],cloud_lb)
        self.output_loss2 = self.criterion_multi(class_logits[1],cloud_lb)
        self.output_loss3 = self.criterion_multi(class_logits[2],cloud_lb)
        self.output_loss4 = self.criterion_multi(class_logits[3],cloud_lb)        
        
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss1 + self.output_loss2 + self.output_loss3 + self.output_loss4 + self.reg_loss    

    def region_mprm_loss(self, cam, regions_all, regions_lb, batch_lengths):
        """
        Runs the overlap region loss on outputs of the model
        :param cam: logits of attention modules (class activation map)
        :param regions_all: indices of subregion points
        :param regions_lb: weak labels of subregions
        :param batch_lengths: batch lengths
        :return: loss
        """        

        # Initilize parameters
        averaged_features = []
        all_cls_lbs = []
        self.output_loss = 0
        cam_all = torch.stack(cam, dim=0)
        star_id = 0

        # Loop over all regions
        for ri in range(len(regions_all)):
            regions = regions_all[ri]
            end_id = star_id + batch_lengths[ri]
            logits = cam_all[:,star_id:end_id,:]
            
            all_cls_lbs.append(np.stack(regions_lb[ri]).astype('float32'))
            for ii in range(len(regions)):
                slc_dix = regions[ii].astype('int64') 
                slc_dix = torch.from_numpy(slc_dix).cuda()
                assert logits.shape[1] >= torch.max(slc_dix), 'logits problem'
                averaged_features.append(torch.mean(logits[:,slc_dix,:], dim=1))

            star_id = star_id + batch_lengths[ri]
        
        all_cls_lbs = np.vstack(all_cls_lbs)
        all_cls_lbs = torch.from_numpy(all_cls_lbs).cuda()
        
        # Stack features and calculate loss
        averaged_features = torch.stack(averaged_features)
        for ii in range(averaged_features.shape[1]):
            self.output_loss = self.output_loss + self.criterion_multi(averaged_features[:,ii,:],all_cls_lbs)

        return self.output_loss 

    def accuracy(self, outputs, labels):        
        # These 2 accuracy functions are definitely sus. I think I only need one of them. 
        # First one is used for pseudo label and second one for weak label script. 
        # Only difference is the dim = 1 / -1. 
        # This only makes a difference when logits dimension is 3D then dim -1 gives the same 
        # results as dim 2 and dim -2 goives the same result as dim 1... 
        # Debug this once pseudo label is running -jer
        """ 
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

    def accuracy_logits(self, logits, labels):
        """
        Computes accuracy of the current batch
        :param logits: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(logits, dim=-1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


class KPFCNN_mprm_ele(KPFCNN_mprm):        
    """
    Class defining KPFCNN for weak labels (multi-path region mining) with elevation attention module
    """

    # Maybe I can just integrate this into the KPFCNN_mprm because I don't need the separation once it runs
    # Maybe this was done by Lin to be able to compare MPRM vs the elevation attention approach -jer

    def __init__(self, config, lbl_values, ign_lbls):
        
        KPFCNN_mprm.__init__(self, config, lbl_values, ign_lbls)

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.features_layer = 'encoder_blocks.5.unary_shortcut.mlp'
        self.forward_features = {}
        self.backward_features = {}

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global', 'attention']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'attention' in block:
                break            
            if 'upsample' in block:
                break
                      
            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config)) # r is radius

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        self.multi_att = dual_att('attention', out_dim, out_dim, r, layer, config)
        self.ele_head = ele_att('ele_attention', 2, out_dim, r, layer, config)          # this is the important part, just to find it quickly later -jer
        self.no_ga = global_average_block('ga1', config.num_classes, config.num_classes, layer, config)
        self.da_ga = global_average_block('ga2', config.num_classes, config.num_classes, layer, config)
        self.spa_ga = global_average_block('ga3', config.num_classes, config.num_classes, layer, config)
        self.cha_ga = global_average_block('ga4', config.num_classes, config.num_classes, layer, config)
        
        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()
        
        # Include elevation attention
        ele_down = batch.points[2][:,-1].unsqueeze_(-1).clone().detach()

        # Loop over consecutive encoder blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        x = self.ele_head(x, ele_down, batch)    
        
        # Combine attention modules and save logits
        spa_att, cha_att, no_att, dual_att = self.multi_att(x, batch)
            
        no_att_cla = self.no_ga(no_att, batch)
        dual_att_cla = self.da_ga(dual_att, batch)
        spa_att_cla = self.spa_ga(spa_att, batch)
        cha_att_cla = self.cha_ga(cha_att, batch)

        cla_logits = [no_att_cla, dual_att_cla, spa_att_cla, cha_att_cla]
               
        # Loop over consecutive decoder blocks
        for block_i, block_op in enumerate(self.decoder_blocks):
            no_att = block_op(no_att, batch)
            dual_att = block_op(dual_att, batch)
            spa_att = block_op(spa_att, batch)
            cha_att = block_op(cha_att, batch)
            
        # Element-wise maximum to get pseudo labels
        x = torch.max(no_att, dual_att)
        x = torch.max(x, spa_att)
        x = torch.max(x, cha_att)
        
        cam = [no_att, dual_att, spa_att, cha_att]

        return x, cla_logits, cam        


















