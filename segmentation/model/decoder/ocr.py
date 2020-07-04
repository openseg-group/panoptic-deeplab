import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["OCR"]


def norm_module(norm, channels):
    if norm == "GN":
        return nn.GroupNorm(32, channels)
    elif norm == "BN":
        return nn.BatchNorm2d(channels)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs, gt_probs=None):
        batch_size, c, h, w = probs.size(0), probs.size(
            1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(
            0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm='BN'):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
<<<<<<< HEAD
                   norm=norm_module(norm, self.key_channels), activation=F.relu_),
            Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
                   norm=norm_module(norm, self.key_channels), activation=F.relu_),
=======
                   norm=norm_module(norm, self.key_channels), activation=F.relu),
            Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
                   norm=norm_module(norm, self.key_channels), activation=F.relu),
>>>>>>> 322282a57f9e1bc3dbd2fe618830ab7361aa5650
        )
        self.f_object = nn.Sequential(
            Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
<<<<<<< HEAD
                   norm=norm_module(norm, self.key_channels), activation=F.relu_),
            Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
                   norm=norm_module(norm, self.key_channels), activation=F.relu_),
        )
        self.f_down = Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                             kernel_size=1, stride=1, padding=0, bias=not norm,
                             norm=norm_module(norm, self.key_channels), activation=F.relu_)
        self.f_up = Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0, bias=not norm,
                           norm=norm_module(norm, self.in_channels), activation=F.relu_)
=======
                   norm=norm_module(norm, self.key_channels), activation=F.relu),
            Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                   kernel_size=1, stride=1, padding=0, bias=not norm,
                   norm=norm_module(norm, self.key_channels), activation=F.relu),
        )
        self.f_down = Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                             kernel_size=1, stride=1, padding=0, bias=not norm,
                             norm=norm_module(norm, self.key_channels), activation=F.relu)
        self.f_up = Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0, bias=not norm,
                           norm=norm_module(norm, self.key_channels), activation=F.relu)
>>>>>>> 322282a57f9e1bc3dbd2fe618830ab7361aa5650
                           
    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(
                h, w), mode='bilinear', align_corners=True)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm='BN'):
        super(ObjectAttentionBlock2D, self).__init__(
            in_channels,
            key_channels,
            scale,
            norm=norm
        )


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 norm='BN'):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels,
            key_channels,
            scale,
            norm
        )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            Conv2d(in_channels=_in_channels, out_channels=out_channels,
                   kernel_size=1, padding=0, bias=not norm,
<<<<<<< HEAD
                   norm=norm_module(norm, out_channels), activation=F.relu_),
=======
                   norm=norm_module(norm, out_channels), activation=F.relu),
>>>>>>> 322282a57f9e1bc3dbd2fe618830ab7361aa5650
            nn.Dropout(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class OCRHead(nn.Module):
    """
    OCR Refine Module.
    """
<<<<<<< HEAD
=======
# self.ocr_head = OCRHead(decoder_channels, decoder_channels, decoder_channels//2, num_classes)
>>>>>>> 322282a57f9e1bc3dbd2fe618830ab7361aa5650
    def __init__(self, in_channels, out_channels, key_channels, num_classes, norm='BN'):
        super().__init__()
        self.num_classes = num_classes
        self.norm = norm
        self.out_channels = out_channels
        self.key_channels = key_channels

        self.gather_head = SpatialGather_Module(num_classes)
        self.distri_head = SpatialOCR_Module(
            in_channels=in_channels,
            key_channels=key_channels,
            out_channels=out_channels,
<<<<<<< HEAD
            scale=2,
=======
            scale=1,
>>>>>>> 322282a57f9e1bc3dbd2fe618830ab7361aa5650
            dropout=0,
            norm=norm
        )

    def forward(self, feat, pred, targets=None):
        context = self.gather_head(feat, pred)
        feat = self.distri_head(feat, context)
        return feat

