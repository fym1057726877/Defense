import math
import torch as th
from torch import nn
from torch.nn import functional as F
from DirectActor.ActorModel import Resnet34Encoder
from gaussiandiffusion import UNetModel


class EncoderAndMemory(nn.Module):
    def __init__(self, feature_dims=4096, MEM_DIM=200, sparse=True):
        super(EncoderAndMemory, self).__init__()

        self.MEM_DIM = MEM_DIM
        self.feature_dims = feature_dims

        self.encoder = ConvEncoder(image_channels=1, conv_channels=64)
        self.decoder = ConvDecoder(image_channels=1, conv_channels=64)

        self.memory = th.zeros((self.MEM_DIM, self.feature_dims))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.sparse = sparse
        self.attn = MultiHeadAttention(
            in_dim=self.feature_dims,
            num_heads=8,
            threshold=1/self.MEM_DIM,
            epsilon=1e-12,
            sparse=self.sparse,
            drop=False,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.encoder(x)  # (B, C, H, W) -> (B, fea)
        assert self.feature_dims == z.shape[1] == C * H * W
        ex_z = z.unsqueeze(1).repeat(1, self.MEM_DIM, 1)  # [b, mem_dim, fea]
        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]

        z_hat, mem_weight = self.attn(ex_z, ex_mem, ex_mem)
        x_recon = self.decoder(z_hat)

        return dict(x_recon=x_recon, mem_weight=mem_weight)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads,
            in_dim,
            embed_dim=None,
            threshold=0.02,
            epsilon=1e-12,
            drop=False,
            droprate=0.1,
            sparse=True,
    ):
        super(MultiHeadAttention, self).__init__()
        embed_dim = embed_dim or in_dim
        assert embed_dim % num_heads == 0  # head_num必须得被embedding_dim整除
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.Cosine_Similiarity = nn.CosineSimilarity(dim=-1)
        # q,k,v个需要一个，最后拼接还需要一个，总共四个线性层
        self.linear_q = nn.Linear(in_dim, self.embed_dim)
        self.linear_k = nn.Linear(in_dim, self.embed_dim)
        self.linear_v = nn.Linear(in_dim, self.embed_dim)

        self.scale = 1 / math.sqrt(self.embed_dim // self.num_heads)

        self.final_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.drop = drop
        if self.drop:
            self.droprate = droprate
            self.dropout = nn.Dropout(p=self.droprate)

        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold
            self.epsilon = epsilon
            self.relu = nn.ReLU(inplace=True)

    def forward(self, q, k, v):
        B, N, L = q.shape
        assert L == self.in_dim, f"{L} != {self.in_dim}"
        # 进入多头处理环节
        # 得到每个头的输入
        dk = self.embed_dim // self.num_heads
        q = self.linear_q(q).view(B, N, self.num_heads, dk).transpose(1, 2)  # B,H,N,dk
        k = self.linear_k(k).view(B, N, self.num_heads, dk).transpose(1, 2)
        v = self.linear_v(v).view(B, N, self.num_heads, dk).transpose(1, 2)

        attn = self.Cosine_Similiarity(q, k).unsqueeze(2)  # B,H,N

        # attn = th.matmul(q, k.transpose(1, 2)) * self.scale  # B,H,N,N
        attn = th.softmax(attn, dim=-1)
        if self.drop:
            attn = self.dropout(attn)

        if self.sparse:
            # 稀疏寻址
            attn = ((self.relu(attn - self.threshold) * attn)
                    / (th.abs(attn - self.threshold) + self.epsilon))
            attn = F.normalize(attn, p=1, dim=1)

        out = th.matmul(attn, v)  # B,H,dk
        out = out.view(B, -1).contiguous()
        out = self.final_linear(out)
        return out, attn.view(B, -1)


class ConvEncoder(nn.Module):
    def __init__(self, image_channels=1, conv_channels=64):
        super(ConvEncoder, self).__init__()
        self.image_channel = image_channels
        self.conv_channel = conv_channels
        self.block1 = self._conv_block(
            in_channels=self.image_channel,
            out_channels=self.conv_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.block2 = self._conv_block(
            in_channels=self.conv_channel // 2,
            out_channels=self.conv_channel,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.block3 = self._conv_block(
            in_channels=self.conv_channel,
            out_channels=self.conv_channel * 2,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.final_conv = nn.Conv2d(
            in_channels=self.conv_channel * 2,
            out_channels=self.conv_channel * 4,
            kernel_size=1,
            stride=2,
            padding=0,
        )

    @staticmethod
    def _conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
    ) -> nn.Sequential():
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_conv(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, image_channels=1, conv_channels=64):
        super(ConvDecoder, self).__init__()
        self.image_channels = image_channels
        self.conv_channels = conv_channels

        self.block1 = self._transconv_block(
            in_channels=self.conv_channels * 4,
            out_channels=self.conv_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.block2 = self._transconv_block(
            in_channels=self.conv_channels * 2,
            out_channels=self.conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.block3 = self._transconv_block(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.final_transconv = nn.ConvTranspose2d(
            in_channels=self.conv_channels // 2,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    @staticmethod
    def _transconv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
            output_padding: int = 0,
    ) -> nn.Sequential():
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # x:[b, self.conv_channel*4*4*4]
        x = x.view(-1, self.conv_channels * 4, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_transconv(x)
        return x


def testencoder():
    e = ConvEncoder(image_channels=1, conv_channels=64)
    x = th.randn((16, 1, 64, 64))
    o = e(x)
    print(o.shape)


def testdecoder():
    d = ConvDecoder(image_channels=1, conv_channels=64)
    x = th.randn((16, 4096))
    o = d(x)
    print(o.shape)


def testmemory():
    model = EncoderAndMemory()
    x = th.randn((16, 1, 64, 64))
    out = model(x)
    x_recon = out
    # x_recon = out["x_recon"]
    # mem_weight = out["mem_weight"]
    print(x_recon.shape)
    # print(mem_weight.shape)


def testattn():
    q = th.randn((16, 200, 4096))
    k = th.randn((16, 200, 4096))
    attn = MultiHeadAttention(num_heads=8, in_dim=4096)
    out = attn(q, k, k)[1]
    print(out.shape)


if __name__ == '__main__':
    # testencoder()
    # testdecoder()
    testmemory()
    # testattn()
