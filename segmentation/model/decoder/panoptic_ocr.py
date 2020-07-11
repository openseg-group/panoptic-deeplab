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


__all__ = ["PanopticOCRDecoder"]


class SinglePanopticOCRDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, num_classes, ocr_channels=None):
        super(SinglePanopticOCRDecoder, self).__init__()
        if ocr_channels is None:
            ocr_channels = decoder_channels

        self.ocr = OCRHead(in_channels, decoder_channels, in_channels//2, num_classes)
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
                fuse_in_channels = ocr_channels + low_level_channels_project[i]
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

    def forward(self, features, coarse_map):
        x = features[self.feature_key]
        x = self.ocr(x, coarse_map)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class PanopticOCRDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes, **kwargs):
        super(PanopticOCRDecoder, self).__init__()
        # Build semantic decoder
        self.feature_key = feature_key

        self.semantic_decoder = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        self.aux_semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['aux_semantic'])

        # extra OCR head
        self.ocr_head = SinglePanopticOCRDecoder(in_channels, feature_key, low_level_channels,
                                                low_level_key, low_level_channels_project,
                                                decoder_channels, num_classes)
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
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(
                decoder_channels=kwargs['instance_decoder_channels'],
                head_channels=kwargs['instance_head_channels'],
                num_classes=kwargs['instance_num_classes'],
                class_key=kwargs['instance_class_key']
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic_features = self.semantic_decoder(features[self.feature_key])
        aux_semantic = self.aux_semantic_head(semantic_features)
        for key in aux_semantic.keys():
            pred[key] = aux_semantic[key]

        # OCR branch
        ocr_features = self.ocr_head(features, aux_semantic['aux_semantic'])
        semantic = self.semantic_head(ocr_features)
        for key in semantic.keys():
            pred[key] = semantic[key]    

        # Instance branch
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]

        return pred
