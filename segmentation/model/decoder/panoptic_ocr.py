# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Rainbowsecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .aspp import ASPP
from .ocr import OCRHead
from .conv_module import stacked_conv, depthwise_separable_conv
from .panoptic_deeplab import SinglePanopticDeepLabDecoder, SinglePanopticDeepLabHead


__all__ = ["PanopticOCRDecoder"]


class OCRCascadeDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, num_classes, ocr_channels=None):
        super(OCRCascadeDecoder, self).__init__()
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

    def set_image_pooling(self, pool_size):
        pass

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


class LightClassHead(nn.Module):
    def __init__(self, decoder_channels, num_classes, class_key):
        super(LightClassHead, self).__init__()

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Conv2d(decoder_channels, num_classes[i], 1)

        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class PanopticOCRDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes, **kwargs):
        super(PanopticOCRDecoder, self).__init__()
        # Build semantic decoder
        self.feature_key = feature_key
        self.low_level_key = low_level_key

        self.bottleneck = depthwise_separable_conv(in_channels + sum(low_level_channels), decoder_channels, 5, padding=1)
        # self.semantic_decoder = nn.Sequential(
        #     nn.Conv2d(in_channels + sum(low_level_channels), decoder_channels, 1, bias=False),
        #     nn.BatchNorm2d(decoder_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aux_semantic_head = SinglePanopticDeepLabHead(in_channels + sum(low_level_channels), decoder_channels, [num_classes], ['aux_semantic'])
        self.aux_semantic_head = LightClassHead(decoder_channels, [num_classes], ['aux_semantic'])

        # extra OCR head
        # self.ocr_head = OCRCascadeDecoder(in_channels, feature_key, low_level_channels,
        #                                   low_level_key, low_level_channels_project,
        #                                   decoder_channels, num_classes)
        self.ocr_head = OCRHead(decoder_channels, decoder_channels, decoder_channels//2, num_classes, scale=2)
        self.semantic_head = LightClassHead(decoder_channels, [num_classes], ['semantic'])

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
                aspp_channels=kwargs['instance_aspp_channels'],
                use_sep_conv=True,
            )
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
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
        # concatenate all the feature maps.
        low_level_features = [features[i] for i in self.low_level_key]
        high_level_feature = features[self.feature_key]
        
        all_level_features = [
            F.interpolate(
                input=x,
                size=low_level_features[-1].shape[2:],
                mode='bilinear',
                align_corners=True) for x in low_level_features
        ]
        high_level_feature = F.interpolate(input=high_level_feature,
                                           size=low_level_features[-1].shape[2:],
                                           mode='bilinear',
                                           align_corners=True)
        
        all_level_features.append(high_level_feature)
        fuse_features = torch.cat(all_level_features, dim=1)

        # print("fuse_features shape: {}".format(fuse_features.shape))

        semantic_features = self.bottleneck(fuse_features)
        aux_semantic = self.aux_semantic_head(semantic_features)

        for key in aux_semantic.keys():
            pred[key] = aux_semantic[key]

        # OCR branch
        ocr_features = self.ocr_head(semantic_features, aux_semantic['aux_semantic'])
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
