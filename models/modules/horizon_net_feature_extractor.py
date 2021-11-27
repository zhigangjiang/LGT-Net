""" 
@author:
@Date: 2021/07/17
@description: Use the feature extractor proposed by HorizonNet
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools
from models.base_model import BaseModule

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''

    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)  # weight padding is 0, LR_PAD then use valid padding will keep dim of weight
        names = name.split('.')

        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


'''
Encoder
'''


class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        features.append(x)  # 1/4
        x = self.encoder.layer2(x)
        features.append(x)  # 1/8
        x = self.encoder.layer3(x)
        features.append(x)  # 1/16
        x = self.encoder.layer4(x)
        features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


'''
Decoder
'''


class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''

    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 4),
            ConvCompressH(in_c // 4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)

        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)  # 先补左右，相当于warp模式，然后进行插值
        d_type = x.dtype
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        # if x.dtype != d_type:
        #     x = x.type(d_type)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1 // out_scale),
            GlobalHeightConv(c2, c2 // out_scale),
            GlobalHeightConv(c3, c3 // out_scale),
            GlobalHeightConv(c4, c4 // out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        # conv_list:
        # 0 [b,  256(d), 128(h), 256(w)] ->(4*{conv3*3 step2*1} : d/8 h/16)-> [b  32(d) 8(h) 256(w)]
        # 1 [b,  512(d),  64(h), 128(w)] ->(4*{conv3*3 step2*1} : d/8 h/16)-> [b  64(d) 4(h) 128(w)]
        # 2 [b, 1024(d),  32(h),  64(w)] ->(4*{conv3*3 step2*1} : d/8 h/16)-> [b 128(d) 2(h)  64(w)]
        # 3 [b, 2048(d),  16(h),  32(w)] ->(4*{conv3*3 step2*1} : d/8 h/16)-> [b 256(d) 1(h)  32(w)]
        # 0 ->(unsampledW256} : w=256)-> [b  32(d) 8(h) 256(w)] ->(reshapeH1} : h=1)-> [b 256(d) 1(h) 256(w)]
        # 1 ->(unsampledW256} : w=256)-> [b  64(d) 4(h) 256(w)] ->(reshapeH1} : h=1)-> [b 256(d) 1(h) 256(w)]
        # 2 ->(unsampledW256} : w=256)-> [b 128(d) 2(h) 256(w)] ->(reshapeH1} : h=1)-> [b 256(d) 1(h) 256(w)]
        # 3 ->(unsampledW256} : w=256)-> [b 256(d) 1(h) 256(w)] ->(reshapeH1} : h=1)-> [b 256(d) 1(h) 256(w)]
        # 0  --\
        # 1  -- \
        #         ---- cat [b 1024(d) 1(h) 256(w)]
        # 2  -- /
        # 3  --/
        return feature  # [b 1024(d) 256(w)]


class HorizonNetFeatureExtractor(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone='resnet50'):
        super(HorizonNetFeatureExtractor, self).__init__()
        self.out_scale = 8
        self.step_cols = 4

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            self.c_last = (c1 * 8 + c2 * 4 + c3 * 2 + c4 * 1) // self.out_scale

        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        x = x.clone()
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        x[:, :3] = (x[:, :3] - self.x_mean) / self.x_std

        return x

    def forward(self, x):
        # x [b 3 512 1024]
        x = self._prepare_x(x)  # [b 3 512 1024]
        conv_list = self.feature_extractor(x)
        # conv_list:
        # 0 [b,  256(d), 128(h), 256(w)]
        # 1 [b,  512(d),  64(h), 128(w)]
        # 2 [b, 1024(d),  32(h),  64(w)]
        # 3 [b, 2048(d),  16(h),  32(w)]
        x = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)  # [b 1024(d) 1(h) 256(w)]
        # After reduce_Height_module, h becomes 1, the information is compressed to d,
        # and w contains different resolutions
        # 0 [b,  256(d), 128(h), 256(w)] -> [b,  256/8(d) * 128/16(h') = 256(d), 1(h) 256(w)]
        # 1 [b,  512(d),  64(h), 128(w)] -> [b,  512/8(d) *  64/16(h') = 256(d), 1(h) 256(w)]
        # 2 [b, 1024(d),  32(h),  64(w)] -> [b, 1024/8(d) *  32/16(h') = 256(d), 1(h) 256(w)]
        # 3 [b, 2048(d),  16(h),  32(w)] -> [b, 2048/8(d) *  16/16(h') = 256(d), 1(h) 256(w)]
        return x  # [b 1024(d) 1(h) 256(w)]


if __name__ == '__main__':
    from PIL import Image
    extractor = HorizonNetFeatureExtractor()
    img = np.array(Image.open("../../src/demo.png")).transpose((2, 0, 1))
    input = torch.Tensor([img])  # 1 3 512 1024
    feature = extractor(input)
    print(feature.shape)  # 1, 1024, 256  |  1024 = (out_c_0*h_0 +... + out_c_3*h_3) = 256 * 4
