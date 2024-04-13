import math
import torch
from torch import nn   
from torch.nn import functional as F

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_layer = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        embed = self.embed_layer(x)
        embed =  embed.permute((0,2,3,1))
        return embed.view(x.shape[0], -1, self.embed_dim)
        # return embed.view(x.shape[0], self.embed_dim, -1)
    

class MlpBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, n_layers=2, activation='gelu', dropout=0.1) -> None:
        super().__init__()
        out_dim = out_dim or in_dim

        match activation:
            case 'relu':
                self.activation = nn.ReLU
            case 'sigmoid':
                self.activation = nn.Sigmoid
            case 'gelu':
                self.activation = nn.GELU

        self.layers = []
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(in_dim, in_dim))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(in_dim, out_dim))
        self.layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*self.layers)
        


    def forward(self, x):
        x = self.layers(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        # assert embed_dim % n_heads == 0
        # self.head_dim = embed_dim // n_heads
        self.head_dim = embed_dim
        
        self.dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(embed_dim, embed_dim* n_heads)
        self.wk = nn.Linear(embed_dim, embed_dim* n_heads)
        self.wv = nn.Linear(embed_dim, embed_dim* n_heads)
        
        self.out_proj = nn.Linear(embed_dim * n_heads, embed_dim)

    def forward(self, q, k=None, v=None):
        k = k if k is not None else q
        v = v if v is not None else q

        q_projs = self.wq(q)
        q_projs = q_projs.tensor_split(self.n_heads, dim=-1)
        k_projs = self.wk(k).tensor_split(self.n_heads, dim=-1)
        v_projs = self.wv(v).tensor_split(self.n_heads, dim=-1)
        
        norm_fact = math.sqrt(self.head_dim)
        contexts = [q@k.transpose(-1, -2) / norm_fact for q, k in zip(q_projs, k_projs)]

        contexts = [self.dropout(F.softmax(c, dim=1)) for c in contexts]

        atts = [c@v for c,v in zip(contexts, v_projs)]

        att = torch.concat(atts, dim=-1)

        return self.out_proj(att)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.self_attention = AttentionBlock(embed_dim, n_heads, dropout)
        self.mlp = MlpBlock(embed_dim,dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        n_x = self.norm1(x)

        x_att = self.self_attention(n_x)
        x = x + x_att

        x_mlp = self.mlp(self.norm2(x))
        return x + x_mlp


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(32,32), patch_size=16, embed_dim=512, n_layers=4, n_heads=8, dropout=0.1, n_classes=None):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        n_patches = (img_size[0]//patch_size) * (img_size[1]//patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches+1, embed_dim))


        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads, 
                    dropout=dropout
                ) for _ in range(n_layers)]
        )

        if n_classes is not None:
            self.head = nn.Linear(embed_dim, n_classes)
        else:
            self.head = nn.Identity()

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        embed = self.patch_embed(x)
        embed = torch.cat([self.class_token.repeat(x.shape[0], 1, 1), embed], dim=1)

        embed += self.pos_embed


        for l in self.layers:
            embed = l(embed)
        

        return self.head(self.out_norm(embed))
        

if __name__ == "__main__":
    embed_layer = PatchEmbedding(16, 128, 3)

    fake_img = torch.randn(16, 3, 224, 224)

    res = embed_layer(fake_img)
    print(res.shape)

    vit = VisionTransformer(n_classes=10, img_size=(224,224))
    res = vit(fake_img)
    print(res.shape)