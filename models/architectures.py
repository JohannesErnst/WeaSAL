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
            x = self.droplayer(x)

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

    def contrast_loss(self, outputs, labels, config, threshold=0.2):
            """
            Runs contrastive loss on outputs of the model by selecting a number of
            points (here slc_con) to compare with each input point and determining
            whether they are negative (of different classes) or positive samples 
            (of the same class). With this metric the contrastive loss is calculated.
            :param outputs: logits predicted by the network
            :param labels: pseudo labels
            :param threshold: threshold for ignoring uncertain labels
            :return: loss
            """

            # Define parameters
            temperature = 0.1
            base_temperature = 1
            self.pts_loss = 0
            self.pts_loss_self = 0
            slc_con = 1000
            N = outputs.shape[0]
            eps = 1e-8
            tensor_0 = torch.tensor(0).float().cuda()
            threshold = config.contrast_thd / 100
            
            # Get probabilities
            prob = torch.nn.Softmax(1)(outputs) 
            pseudo_logits = prob.max(1)[0]

            # Gather all labeled points and confident unlabeled points
            # (label > 10 means the point is unlabeled)
            label_id = labels < 10
            certain_label = pseudo_logits > threshold
            certain_label = (certain_label + label_id) > 0

            # Update unlabeled points with confident predictions (argmax) and save the valid indices
            pseudo_lbs = torch.argmax(prob, dim=1)
            pseudo_lbs[label_id] = labels[label_id] 
            all_valid_idx = torch.where(certain_label)[0]

            # Collect slc_con random indices from the valid pseudo labels (--> slice)
            num_valid = all_valid_idx.shape[0] 
            if num_valid < 1:
                print('Skipped loss calculations because there are no valid points in batch')
                return tensor_0
            if num_valid >= slc_con:
                slc_idx_idx = torch.randint(0,num_valid,(slc_con,))
                slc_idx = all_valid_idx[slc_idx_idx]
            else:
                o_idx = torch.arange(num_valid)
                slc_idx_idx =  torch.randint(0,num_valid,(slc_con-num_valid,))
                slc_idx_idx = torch.cat((o_idx, slc_idx_idx), dim = 0)
                slc_idx = all_valid_idx[slc_idx_idx]

            # Create a mask [N x slc_con] that marks all slice indices per input point
            mask_slice = torch.ones(N, slc_con)
            all_idx = torch.arange(N).cuda()
            idx = torch.where(all_idx[:, None] == slc_idx[None, :])
            cc = torch.stack(idx, dim=0).T
            cc = cc.cuda()
            mask_slice[cc[:,0], cc[:,1]] = 0
            mask_slice = mask_slice.cuda()
                    
            # Collect pseudo labels and points of the slice
            pseudo_label_slc = pseudo_lbs[slc_idx]
            certain_label_slc = certain_label[slc_idx]
            
            # Create a mask [N x slc_con] that marks all certain label indices per input point
            mask_certain = certain_label_slc.unsqueeze(0) == certain_label.unsqueeze(-1) 
            
            # Create a mask [N x slc_con] that marks where the slice point labels are equal 
            # to the input point labels (i.e. the so called positive samples in paper)
            mask_positive = pseudo_label_slc.unsqueeze(0) == pseudo_lbs.unsqueeze(-1)

            # Concatenate masks to get the positions of the points used for calculating 
            # the supervised contrastive loss per input point
            pos_mask = mask_positive * mask_slice * mask_certain

            # Select normalized slice-logits as [N x slc_con] and weight with temperature
            outputs = nn.functional.normalize(outputs, dim=1)
            x_slc = outputs[slc_idx]
            mul = torch.div(torch.matmul(outputs, x_slc.T), temperature)

            # For numerical stability
            logits_max, _ = torch.max(mul, dim=1, keepdim=True)
            logits = mul - logits_max.detach()

            # Compute logarithmic probability
            exp_logits = torch.exp(logits) * (mask_slice * mask_certain)
            log_prob = (logits - torch.log((exp_logits.sum(1, keepdim=True)+eps))) * (mask_slice * mask_certain) 

            # Compute mean of log-likelihood over positive samples for each point.
            # If no positve samples are found, take them as 0
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1)+1e-12)
            self.pts_loss = - (temperature / base_temperature) * mean_log_prob_pos
            cal_slc = self.pts_loss > 0
            self.pts_loss = self.pts_loss[cal_slc]
            pseudo_lbs = pseudo_lbs[cal_slc]
            self.pts_loss = scatter(self.pts_loss, pseudo_lbs, reduce="mean")
            self.pts_loss = self.pts_loss[self.pts_loss>0]

            return self.pts_loss.mean()


class KPFCNN_mprm(nn.Module):
    """
    Class defining KPFCNN for weak labels with multi-path region mining (mprm) 
    and elevation attention module.
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
        
        self.multi_att = multi_path_att('attention', out_dim, out_dim, r, layer, config)
        self.ele_head = ele_att('ele_attention', 2, out_dim, r, layer, config)
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

        # Choose segmentation loss
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
        spa_att, cha_att, no_att, poi_att = self.multi_att(x, batch)

        # Perform global average pooling
        no_att_cla = self.no_ga(no_att, batch)
        poi_att_cla = self.da_ga(poi_att, batch)
        spa_att_cla = self.spa_ga(spa_att, batch)
        cha_att_cla = self.cha_ga(cha_att, batch)  
        cla_logits = [no_att_cla, poi_att_cla, spa_att_cla, cha_att_cla]
        
        # Loop over consecutive decoder blocks
        for block_i, block_op in enumerate(self.decoder_blocks):
            no_att = block_op(no_att, batch)
            poi_att = block_op(poi_att, batch)
            spa_att = block_op(spa_att, batch)
            cha_att = block_op(cha_att, batch)

        # Element-wise maximum to get pseudo labels
        x = torch.max(no_att, poi_att)
        x = torch.max(x, spa_att)
        x = torch.max(x, cha_att)
            
        # Grab point class activation maps
        cam = [no_att, poi_att, spa_att, cha_att]
        
        return x, cla_logits, cam
   
    def class_logits_loss(self, class_logits, cloud_lb):
        """
        Runs the BCEWithLogitsLoss (binary cross entropy) on outputs of the model.
        Uses one weak labels per input sphere (and not per subclouds).
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
        Runs the overlap region loss on outputs of the model.
        Uses weak labels per subcloud (so multiple weak labels in one input sphere).
        :param cam: logits of attention modules (class activation map)
        :param regions_all: indices of subregion points
        :param regions_lb: ground truth weak labels of subregions
        :param batch_lengths: batch lengths
        :return: loss
        """        

        # Initilize parameters
        averaged_features = []
        all_cls_lbs = []
        self.output_loss = 0
        cam_all = torch.stack(cam, dim=0)
        start_id = 0

        # Loop over all regions (input spheres)
        for ri in range(len(regions_all)):

            # Determine correct end index of input sphere from batch lengths
            end_id = start_id + batch_lengths[ri]

            # Check if input sphere has subclouds (important for active learning)
            if regions_all[ri]:
                regions = regions_all[ri]
                logits = cam_all[:,start_id:end_id,:]

                # Retrieve ground-truth weak labels for all subspheres within the whole input sphere
                all_cls_lbs.append(np.stack(regions_lb[ri]).astype('float32'))

                # Retrieve weak labels of each subregion based on output logits (predicted)
                for subregion in range(len(regions)):
                    slc_dix = regions[subregion].astype('int64')
                    slc_dix = torch.from_numpy(slc_dix).cuda()
                    assert logits.shape[1] >= torch.max(slc_dix), 'logits problem'
                    averaged_features.append(torch.mean(logits[:,slc_dix,:], dim=1))

            # Update start index for next input sphere
            start_id = start_id + batch_lengths[ri]
        
        # Stack weak labels (ground truth and predicted) and calculate loss
        all_cls_lbs = np.vstack(all_cls_lbs)
        all_cls_lbs = torch.from_numpy(all_cls_lbs).cuda()
        averaged_features = torch.stack(averaged_features)
        for ii in range(averaged_features.shape[1]):
            self.output_loss = self.output_loss + self.criterion_multi(averaged_features[:,ii,:],all_cls_lbs)

        return self.output_loss

    def accuracy(self, logits, labels):        
        """ 
        Computes accuracy of the current batch
        :param logits: logits output predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Check dimension
        if not len(logits.size()) == 2:
            raise ValueError('Wrong logits output dimension: Expected 2, got ' + str(len(logits.size())))

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(logits, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
