# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseSegmentationModel
from segmentation.model.decoder import PanopticFCNDecoder
from segmentation.utils import AverageMeter


__all__ = ["PanopticFCN"]


# class PanopticFCNDecoder(nn.Module):
#     def __init__(self, 
#                  in_channels, 
#                  feature_key, 
#                  low_level_channels, 
#                  low_level_key, 
#                  low_level_channels_project,
#                  decoder_channels, 
#                  atrous_rates,
#                  num_classes, 
#                  **kwargs):

class PanopticFCN(BaseSegmentationModel):
    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key,
                 low_level_channels_project, decoder_channels, atrous_rates, num_classes,
                 semantic_loss, semantic_loss_weight, center_loss, center_loss_weight,
                 offset_loss, offset_loss_weight, **kwargs):
        decoder = PanopticFCNDecoder(in_channels, feature_key, low_level_channels, low_level_key,
                                     low_level_channels_project, decoder_channels, atrous_rates, num_classes, **kwargs)
        super(PanopticFCN, self).__init__(backbone, decoder)

        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()
        self.loss_meter_dict['Semantic loss'] = AverageMeter()

        if kwargs.get('has_instance', False):
            self.center_loss = center_loss
            self.center_loss_weight = center_loss_weight
            self.offset_loss = offset_loss
            self.offset_loss_weight = offset_loss_weight
            self.loss_meter_dict['Center loss'] = AverageMeter()
            self.loss_meter_dict['Offset loss'] = AverageMeter()
        else:
            self.center_loss = None
            self.center_loss_weight = 0
            self.offset_loss = None
            self.offset_loss_weight = 0

        # Initialize parameters.
        self._init_params()

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        loss = 0
        if targets is not None:
            if 'semantic_weights' in targets.keys():
                semantic_loss = self.semantic_loss(
                    results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']
                ) * self.semantic_loss_weight
            else:
                semantic_loss = self.semantic_loss(
                    results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            loss += semantic_loss
            if self.center_loss is not None:
                # Pixel-wise loss weight
                center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
                center_loss = self.center_loss(results['center'], targets['center']) * center_loss_weights
                # safe division
                if center_loss_weights.sum() > 0:
                    center_loss = center_loss.sum() / center_loss_weights.sum() * self.center_loss_weight
                else:
                    center_loss = center_loss.sum() * 0
                self.loss_meter_dict['Center loss'].update(center_loss.detach().cpu().item(), batch_size)
                loss += center_loss
            if self.offset_loss is not None:
                # Pixel-wise loss weight
                offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
                offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
                # safe division
                if offset_loss_weights.sum() > 0:
                    offset_loss = offset_loss.sum() / offset_loss_weights.sum() * self.offset_loss_weight
                else:
                    offset_loss = offset_loss.sum() * 0
                self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
                loss += offset_loss
        # In distributed DataParallel, this is the loss on one machine, need to average the loss again
        # in train loop.
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)
        return results
