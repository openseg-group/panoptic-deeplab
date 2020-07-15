# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .aspp import ASPP
from .ocr import OCRHead
from .conv_module import stacked_conv
from .panoptic_deeplab import SinglePanopticDeepLabDecoder, SinglePanopticDeepLabHead


__all__ = ["PanopticFCNDecoder"]


class SinglePanopticFCNDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 feature_key, 
                 low_level_channels, 
                 low_level_key, 
                 low_level_channels_project,
                 decoder_channels,
                 atrous_rates, 
                 aspp_channels=None):
        super(SinglePanopticFCNDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels

        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU(inplace=True)
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        pass

    def forward(self, features):
        x = features[self.feature_key]
        x = self.fcn(x)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class PanopticFCNDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 feature_key, 
                 low_level_channels, 
                 low_level_key, 
                 low_level_channels_project,
                 decoder_channels, 
                 atrous_rates,
                 num_classes, 
                 **kwargs):
        super(PanopticFCNDecoder, self).__init__()
        # Build semantic decoder
        self.feature_key = feature_key

        self.semantic_decoder = SinglePanopticFCNDecoder(in_channels, feature_key, low_level_channels,
                                                         low_level_key, low_level_channels_project,
                                                         decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])

        # Build instance decoder
        self.instance_decoder = None
        self.instance_head = None
        if kwargs.get('has_instance', False):
            instance_decoder_kwargs = dict(
                in_channels=in_channels,
                feature_key=feature_key,
                low_level_channels=low_level_channels,
                low_level_key=low_level_key,
                low_level_channels_project=kwargs['instance_low_level_channels_project'],
                decoder_channels=kwargs['instance_decoder_channels'],
                atrous_rates=atrous_rates,
                aspp_channels=kwargs['instance_aspp_channels']
            )
            self.instance_decoder = SinglePanopticFCNDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(
                decoder_channels=kwargs['instance_decoder_channels'],
                head_channels=kwargs['instance_head_channels'],
                num_classes=kwargs['instance_num_classes'],
                class_key=kwargs['instance_class_key']
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_image_pooling(self, pool_size):
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Instance branch
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]

        return pred
