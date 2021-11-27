from models.modules.transformer_modules import *


class SWG_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, win_size, dim_head, mlp_dim,
                 dropout=0., patch_num=None, ape=None, rpe=None, rpe_pos=1):
        super().__init__()
        self.absolute_pos_embed = None if patch_num is None or ape is None else AbsolutePosition(dim, dropout,
                                                                                                 patch_num, ape)
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i % 2 == 0:
                attention = WinAttention(dim, win_size=win_size, shift=0 if (i % 3 == 0) else win_size // 2,
                                         heads=heads, dim_head=dim_head, dropout=dropout, rpe=rpe, rpe_pos=rpe_pos)
            else:
                attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                      patch_num=patch_num, rpe=rpe, rpe_pos=rpe_pos)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        if self.absolute_pos_embed is not None:
            x = self.absolute_pos_embed(x)
        x = self.pos_dropout(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


if __name__ == '__main__':
    token_dim = 1024
    toke_len = 256

    transformer = SWG_Transformer(dim=token_dim,
                                  depth=6,
                                  heads=16,
                                  win_size=8,
                                  dim_head=64,
                                  mlp_dim=2048,
                                  dropout=0.1)

    input = torch.randn(1, toke_len, token_dim)
    output = transformer(input)
    print(output.shape)
