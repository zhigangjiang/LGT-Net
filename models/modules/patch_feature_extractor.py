import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchFeatureExtractor(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, patch_num=256, input_shape=None):
        super(PatchFeatureExtractor, self).__init__()

        if input_shape is None:
            input_shape = [3, 512, 1024]
        self.patch_dim = 1024
        self.patch_num = patch_num

        img_channel = input_shape[0]
        img_h = input_shape[1]
        img_w = input_shape[2]

        p_h, p_w = img_h, img_w // self.patch_num
        p_dim = p_h * p_w * img_channel

        self.patch_embedding = nn.Sequential(
            Rearrange('b c h (p_n p_w) -> b p_n (h p_w c)', p_w=p_w),
            nn.Linear(p_dim, self.patch_dim)
        )

        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False

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
        x = self.patch_embedding(x)  # [b 256(patch_num) 1024(d)]
        x = x.permute(0, 2, 1)  # [b 1024(d) 256(patch_num)]
        return x


if __name__ == '__main__':
    from PIL import Image
    extractor = PatchFeatureExtractor()
    img = np.array(Image.open("../../src/demo.png")).transpose((2, 0, 1))
    input = torch.Tensor([img])  # 1 3 512 1024
    feature = extractor(input)
    print(feature.shape)  # 1, 1024, 256
