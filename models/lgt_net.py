import torch.nn
import torch
import torch.nn as nn
import models.modules as modules
import numpy as np

from models.base_model import BaseModule
from models.modules.horizon_net_feature_extractor import HorizonNetFeatureExtractor
from models.modules.patch_feature_extractor import PatchFeatureExtractor
from utils.conversion import uv2depth, get_u, lonlat2depth, get_lon, lonlat2uv
from utils.height import calc_ceil_ratio
from utils.misc import tensor2np


class LGT_Net(BaseModule):
    def __init__(self, ckpt_dir=None, backbone='resnet50', dropout=0.0, output_name='LGT',
                 decoder_name='Transformer', win_size=8, depth=6,
                 ape=None, rpe=None, corner_heat_map=False, rpe_pos=1):
        super().__init__(ckpt_dir)

        self.patch_num = 256
        self.patch_dim = 1024
        self.decoder_name = decoder_name
        self.output_name = output_name
        self.corner_heat_map = corner_heat_map
        self.dropout_d = dropout

        if backbone == 'patch':
            self.feature_extractor = PatchFeatureExtractor(patch_num=self.patch_num, input_shape=[3, 512, 1024])
        else:
            # feature extractor
            self.feature_extractor = HorizonNetFeatureExtractor(backbone)

        if 'Transformer' in self.decoder_name:
            # transformer encoder
            transformer_dim = self.patch_dim
            transformer_layers = depth
            transformer_heads = 8
            transformer_head_dim = transformer_dim // transformer_heads
            transformer_ff_dim = 2048
            rpe = None if rpe == 'None' else rpe
            self.transformer = getattr(modules, decoder_name)(dim=transformer_dim, depth=transformer_layers,
                                                              heads=transformer_heads, dim_head=transformer_head_dim,
                                                              mlp_dim=transformer_ff_dim, win_size=win_size,
                                                              dropout=self.dropout_d, patch_num=self.patch_num,
                                                              ape=ape, rpe=rpe, rpe_pos=rpe_pos)
        elif self.decoder_name == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=self.feature_extractor.c_last,
                                  hidden_size=self.patch_dim // 2,
                                  num_layers=2,
                                  dropout=self.dropout_d,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(self.dropout_d)
        else:
            raise NotImplementedError("Only support *Transformer and LSTM")

        if self.output_name == 'LGT':
            # omnidirectional-geometry aware output
            self.linear_depth_output = nn.Linear(in_features=self.patch_dim, out_features=1)
            self.linear_ratio = nn.Linear(in_features=self.patch_dim, out_features=1)
            self.linear_ratio_output = nn.Linear(in_features=self.patch_num, out_features=1)
        elif self.output_name == 'LED' or self.output_name == 'Horizon':
            # horizon-depth or latitude output
            self.linear = nn.Linear(in_features=self.patch_dim, out_features=2)
        else:
            raise NotImplementedError("Unknown output")

        if self.corner_heat_map:
            # corners heat map output
            self.linear_corner_heat_map_output = nn.Linear(in_features=self.patch_dim, out_features=1)

        self.name = f"{self.decoder_name}_{self.output_name}_Net"

    def lgt_output(self, x):
        """
        :param x: [ b, 256(patch_num), 1024(d)]
        :return: {
            'depth': [b, 256(patch_num & d)]
            'ratio': [b, 1(d)]
        }
        """
        depth = self.linear_depth_output(x)  # [b, 256(patch_num), 1(d)]
        depth = depth.view(-1, self.patch_num)  # [b, 256(patch_num & d)]

        # ratio represent room height
        ratio = self.linear_ratio(x)  # [b, 256(patch_num), 1(d)]
        ratio = ratio.view(-1, self.patch_num)  # [b, 256(patch_num & d)]
        ratio = self.linear_ratio_output(ratio)  # [b, 1(d)]
        output = {
            'depth': depth,
            'ratio': ratio
        }
        return output

    def led_output(self, x):
        """
        :param x: [ b, 256(patch_num), 1024(d)]
        :return: {
            'depth':  [b, 256(patch_num)]
            'ceil_depth': [b, 256(patch_num)]
            'ratio': [b, 1(d)]
        }
        """
        bon = self.linear(x)  # [b, 256(patch_num), 2(d)]
        bon = bon.permute(0, 2, 1)  # [b, 2(d), 256(patch_num)]
        bon = torch.sigmoid(bon)

        ceil_v = bon[:, 0, :] * -0.5 + 0.5  # [b, 256(patch_num)]
        floor_v = bon[:, 1, :] * 0.5 + 0.5  # [b, 256(patch_num)]
        u = get_u(w=self.patch_num, is_np=False, b=ceil_v.shape[0]).to(ceil_v.device)
        ceil_boundary = torch.stack((u, ceil_v), axis=-1)  # [b, 256(patch_num), 2]
        floor_boundary = torch.stack((u, floor_v), axis=-1)  # [b, 256(patch_num), 2]
        output = {
            'depth': uv2depth(floor_boundary),  # [b, 256(patch_num)]
            'ceil_depth': uv2depth(ceil_boundary),  # [b, 256(patch_num)]
        }
        # print(output['depth'].mean())
        if not self.training:
            # [b, 1(d)]
            output['ratio'] = calc_ceil_ratio([tensor2np(ceil_boundary), tensor2np(floor_boundary)], mode='lsq').reshape(-1, 1)
        return output

    def horizon_output(self, x):
        """
        :param x: [ b, 256(patch_num), 1024(d)]
        :return: {
            'floor_boundary':  [b, 256(patch_num)]
            'ceil_boundary': [b, 256(patch_num)]
        }
        """
        bon = self.linear(x)  # [b, 256(patch_num), 2(d)]
        bon = bon.permute(0, 2, 1)  # [b, 2(d), 256(patch_num)]

        output = {
            'boundary': bon
        }
        if not self.training:
            lon = get_lon(w=self.patch_num, is_np=False, b=bon.shape[0]).to(bon.device)
            floor_lat = torch.clip(bon[:, 0, :], 1e-4, np.pi / 2)
            ceil_lat = torch.clip(bon[:, 1, :], -np.pi / 2, -1e-4)
            floor_lonlat = torch.stack((lon, floor_lat), axis=-1)  # [b, 256(patch_num), 2]
            ceil_lonlat = torch.stack((lon, ceil_lat), axis=-1)  # [b, 256(patch_num), 2]
            output['depth'] = lonlat2depth(floor_lonlat)
            output['ratio'] = calc_ceil_ratio([tensor2np(lonlat2uv(ceil_lonlat)),
                                               tensor2np(lonlat2uv(floor_lonlat))], mode='mean').reshape(-1, 1)
        return output

    def forward(self, x):
        """
        :param x: [b, 3(d), 512(h), 1024(w)]
        :return: {
            'depth': [b, 256(patch_num & d)]
            'ratio': [b, 1(d)]
        }
        """

        # feature extractor
        x = self.feature_extractor(x)  # [b 1024(d) 256(w)]

        if 'Transformer' in self.decoder_name:
            # transformer decoder
            x = x.permute(0, 2, 1)  # [b 256(patch_num) 1024(d)]
            x = self.transformer(x)  # [b 256(patch_num) 1024(d)]
        elif self.decoder_name == 'LSTM':
            # lstm decoder
            x = x.permute(2, 0, 1)  # [256(patch_num), b, 1024(d)]
            self.bi_rnn.flatten_parameters()
            x, _ = self.bi_rnn(x)  # [256(patch_num & seq_len), b, 1024(d)]
            x = x.permute(1, 0, 2)  # [b, 256(patch_num), 1024(d)]
            x = self.drop_out(x)

        output = None
        if self.output_name == 'LGT':
            # plt output
            output = self.lgt_output(x)

        elif self.output_name == 'LED':
            # led output
            output = self.led_output(x)

        elif self.output_name == 'Horizon':
            # led output
            output = self.horizon_output(x)

        if self.corner_heat_map:
            corner_heat_map = self.linear_corner_heat_map_output(x)  # [b, 256(patch_num), 1]
            corner_heat_map = corner_heat_map.view(-1, self.patch_num)
            corner_heat_map = torch.sigmoid(corner_heat_map)
            output['corner_heat_map'] = corner_heat_map

        return output


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from models.other.init_env import init_env

    init_env(0, deterministic=True)

    net = LGT_Net()

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('parameter total:{:,}, trainable:{:,}'.format(total, trainable))

    img = np.array(Image.open("../src/demo.png")).transpose((2, 0, 1))
    input = torch.Tensor([img])  # 1 3 512 1024
    output = net(input)

    print(output['depth'].shape)  # 1 256
    print(output['ratio'].shape)  # 1 1
