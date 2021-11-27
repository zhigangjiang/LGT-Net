import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class Attend(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, dim=self.dim, dtype=input.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Conv(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        x = self.net(x)
        return x.transpose(1, 2)


class ConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, Conv(dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff, cov in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            x = cov(x) + x
        return x


if __name__ == '__main__':
    token_dim = 1024
    toke_len = 256

    transformer = ConvTransformer(dim=token_dim,
                                  depth=6,
                                  heads=16,
                                  dim_head=64,
                                  mlp_dim=2048,
                                  dropout=0.1)

    total = sum(p.numel() for p in transformer.parameters())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print('parameter total:{:,}, trainable:{:,}'.format(total, trainable))

    input = torch.randn(1, toke_len, token_dim)
    output = transformer(input)
    print(output.shape)
