## from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): #注意力机制
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) #把dim映射到后者

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        # 计算Q (Query), K (Key), V (Value)。这三者都来自输入x。
        # 使用self.to_qkv(x)进行变换后，通过.chunk(3, dim=-1)分割得到Q, K, V三个部分。
        qkv = self.to_qkv(x).chunk(3, dim=-1)#1 是batchsize也就是一个batch一张图片，196个pathc+一个cls符号，1024是
        #我每一个输入tokenembedding的维度
        ## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        # 使用map和lambda函数对Q, K, V进行重整，将它们分为多个头。
        # 'b' 是batch size, 'h' 是头的数量, 'n' 是token数量 (如补丁数量+CLS token), 'd' 是每个头的维度。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算Q和K的点积，并进行缩放。
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 使用self.attend函数计算注意力权重。
        attn = self.attend(dots)

        # 使用计算出的注意力权重对V进行加权求和。
        out = torch.matmul(attn, v)

        # 重组输出，以合并多头的结果。
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 使用self.to_out对结果进行线性变换并返回。
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #这里的prenorm表示在xxx之前进行一个归一化
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)  ## 224*224
        patch_height, patch_width = pair(patch_size)  ## 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) #可以分割成多少个
        patch_dim = channels * patch_height * patch_width #然后要将patch拉平

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), #修改张亮的维度
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  ## img 1 3 224 224  输出形状x : 1 196 1024 为什么是1024呢，在我重排图像（展平之后，我还会线性映射到dim）
        b, n, _ = x.shape  ##b:batchsize n:length

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) #copy b份 每个都要加个cls
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


v = ViT(
    image_size=224,#the size of images
    patch_size=16,#each patch size
    num_classes=1000,#做多少类别的分类
    dim=1024,#维度
    depth=6,#encoder堆叠多少个
    heads=16,#多头--打在多少个子空间上
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 224, 224)

preds = v(img)  # (1, 1000)