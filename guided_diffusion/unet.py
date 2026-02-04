from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .sem_vis_dual import DualCrossAttention

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)



class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))


    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


































































































































































































































import torch
import torch.nn as nn

class CrossAttnBlock(nn.Module):
    def __init__(self, dim_q, dim_ctx, heads=8, dim_head=64, p_drop=0.0, pre_norm=True, use_cosine=True, tau_init=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.pre_norm = pre_norm
        self.use_cosine = use_cosine

        self.ln_q   = nn.LayerNorm(dim_q)
        self.ln_ctx = nn.LayerNorm(dim_ctx)

        self.q_proj = nn.Linear(dim_q,  inner, bias=False)
        self.k_proj = nn.Linear(dim_ctx, inner, bias=False)
        self.v_proj = nn.Linear(dim_ctx, inner, bias=False)

        self.out = nn.Linear(inner, dim_q, bias=False)
        self.drop = nn.Dropout(p_drop)
        self.scale = dim_head ** -0.5
        if tau_init is None:
            tau_init = (dim_head ** -0.5)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

    def forward(self, x, context, key_padding_mask=None):
        B, Lq, Dq = x.shape
        Bc, Lk, Dc = context.shape
        assert B == Bc

        q_in  = self.ln_q(x)   if self.pre_norm else x
        kv_in = self.ln_ctx(context) if self.pre_norm else context

        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        H, Dh = self.heads, self.dim_head
        q = q.view(B, Lq, H, Dh).transpose(1, 2)
        k = k.view(B, Lk, H, Dh).transpose(1, 2)
        v = v.view(B, Lk, H, Dh).transpose(1, 2)

        if self.use_cosine:
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            attn_scores = (q @ k.transpose(-2, -1)) / (self.tau + 1e-6)
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = attn_scores.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Lq, H*Dh)
        out = self.out(out)
        return out


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _infer_hw_from_n(n: int):
    r = int(round(math.sqrt(n)))
    return (r, r) if r * r == n else None

class LearnableTokenResizer(nn.Module):
    """
    把 token 序列 (B, N, C) 学习式地重采样到目标空间 (H, W)，返回 (B, H*W, C)。
    逻辑：
      1) 先把 N 还原成近方阵 H0×W0（必要时 pad）
      2) 用一串 ConvTranspose2d/Conv2d 把 H0×W0 变为 H×W
         - 先做若干次 x2 的上/下采样（stride=2）
         - 剩余的“小差值”用 stride=1 的 conv/convtranspose 一步到位（通过 kernel 控制尺寸）
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.paths = nn.ModuleDict()

    def _make_path(self, H0, W0, H, W):
        path = []

        def up2(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )

        def down2(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )

        curH, curW = H0, W0
        C = self.dim
        while (curH * 2 <= H) and (curW * 2 <= W):
            path.append(up2(C, C))
            curH, curW = curH * 2, curW * 2

        while (curH % 2 == 0) and (curH // 2 >= H) and (curW % 2 == 0) and (curW // 2 >= W):
            path.append(down2(C, C))
            curH, curW = curH // 2, curW // 2



        def add_delta(cur, target, is_transpose):
            delta = target - cur
            if delta == 0:
                return None
            if is_transpose:

                k = int(target - cur + 1)
                p = 0
                return k, p
            else:

                k = int(cur - target + 1)
                p = 0
                return k, p
        if curH != H:
            is_up = H > curH
            kH, pH = add_delta(curH, H, is_transpose=is_up)
        else:
            kH, pH = None, None
        if curW != W:
            is_up = W > curW
            kW, pW = add_delta(curW, W, is_transpose=is_up)
        else:
            kW, pW = None, None
        if curH != H:
            if H > curH:
                path.append(nn.ConvTranspose2d(C, C, kernel_size=(kH, 1), stride=1, padding=(pH, 0)))
            else:
                path.append(nn.Conv2d(C, C, kernel_size=(kH, 1), stride=1, padding=(pH, 0)))
            path.append(nn.GELU())
            curH = H

        if curW != W:
            if W > curW:
                path.append(nn.ConvTranspose2d(C, C, kernel_size=(1, kW), stride=1, padding=(0, pW)))
            else:
                path.append(nn.Conv2d(C, C, kernel_size=(1, kW), stride=1, padding=(0, pW)))
            path.append(nn.GELU())
            curW = W

        return nn.Sequential(*path)

    def forward(self, tokens_bnc, target_h, target_w):
        B, N, C = tokens_bnc.shape
        assert C == self.dim, f"channel mismatch: {C} vs {self.dim}"

        hw = _infer_hw_from_n(N)
        if hw is None:
            side = int(round(math.sqrt(N)))
            need = side * side - N
            if need > 0:
                pad = tokens_bnc[:, -1:, :].expand(B, need, C)
                tokens_bnc = torch.cat([tokens_bnc, pad], dim=1)
            H0, W0 = side, side
        else:
            H0, W0 = hw

        feat = tokens_bnc[:, :H0*W0, :].reshape(B, H0, W0, C).permute(0, 3, 1, 2)

        key = f"{H0}x{W0}->{target_h}x{target_w}"
        if key not in self.paths:
            self.paths[key] = self._make_path(H0, W0, target_h, target_w).to("cuda:1")

        feat = self.paths[key](feat)
        return feat.permute(0, 2, 3, 1).reshape(B, target_h * target_w, C)




class HCR(nn.Module):
    def __init__(self, dim_q, dim_s=512, dim_v=512, heads=8, dim_head=64, alpha_init=0.5, lambda_init=0.3, p_drop=0.0):
        super().__init__()
        self.ca_s = CrossAttnBlock(dim_q=dim_q, dim_ctx=dim_s, heads=heads, dim_head=dim_head, p_drop=p_drop, pre_norm=True, use_cosine=True)
        self.ca_v = CrossAttnBlock(dim_q=dim_q, dim_ctx=dim_v, heads=heads, dim_head=dim_head, p_drop=p_drop, pre_norm=True, use_cosine=True)
        self.feat_ln = nn.LayerNorm(dim_q)
        self.alpha_s = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_v = nn.Parameter(torch.tensor(1.0 - alpha_init))
        self.lambda_scale = nn.Parameter(torch.tensor(lambda_init))
        self.v_resizer = LearnableTokenResizer(dim=dim_v)


    def forward(self, unet_feat_bchw, S_bnc, V_bnc, txt_pad_mask=None, img_patch_mask=None):
        B, C, H, W = unet_feat_bchw.shape
        feat = unet_feat_bchw.flatten(2).transpose(1, 2)
        feat = self.feat_ln(feat)















        return feat.transpose(1, 2).reshape(B, C, H, W)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1





















        self.dca = DualCrossAttention(d_txt=512, d_img=768, d_model=512, n_heads=8, p_dropout=0.1)







        self.attention_module1 = MultiHeadAttention(embed_dim=512, num_heads=8)
        self.attention_module2 = MultiHeadAttention(embed_dim=512, num_heads=8)
        self.attention_module3 = MultiHeadAttention(embed_dim=512, num_heads=8)













        self.mix_tau = 1.0

        self.mix1 = nn.Conv2d(1024, 2, kernel_size=1, bias=True)
        self.mix2 = nn.Conv2d(1024, 2, kernel_size=1, bias=True)
        self.mix3 = nn.Conv2d(2048, 2, kernel_size=1, bias=True)
        self.resgate1 = nn.Conv2d(512, 1, kernel_size=1, bias=True)
        self.resgate2 = nn.Conv2d(512, 1, kernel_size=1, bias=True)
        self.resgate3 = nn.Conv2d(1024, 1, kernel_size=1, bias=True)


        for m in [self.mix1, self.mix2, self.mix3]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        p0 = 0.1
        init_b = math.log(p0) - math.log(1 - p0)
        for g in [self.resgate1, self.resgate2, self.resgate3]:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, init_b)
        self.channel_adjust = nn.Conv2d(1024,512, kernel_size=1)
        self.channel_adjust1 = nn.Conv2d(512,1024, kernel_size=1)
        self.channel_adjust2 = nn.Conv2d(512,1024, kernel_size=1)
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.dca.apply(convert_module_to_f16)
        self.hcr_64.apply(convert_module_to_f16)
        self.hcr_32.apply(convert_module_to_f16)
        self.hcr_16.apply(convert_module_to_f16)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.dca.apply(convert_module_to_f32)
        self.hcr_64.apply(convert_module_to_f32)
        self.hcr_32.apply(convert_module_to_f32)
        self.hcr_16.apply(convert_module_to_f32)
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self,x,timesteps, text_seq,img_seq_no_cls,txt_pad_mask,y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        s, v = self.dca(
            text_seq=text_seq,
            img_seq=img_seq_no_cls,
            txt_key_padding_mask=txt_pad_mask,
            img_key_padding_mask=None
        )
        print("s.shape:",s.shape)
        print("v.shape:",v.shape)



        print("model forward t:",timesteps)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        print("------------------------")
        print("emb.shape:",emb.shape)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)


        print("----input_blocks----")
        input_sum=0
        for module in self.input_blocks:
            input_sum=input_sum+1
            h = module(h, emb)
            print("input_sum:",input_sum)

            max_value = th.max(h)
            print("最大值:", max_value)
            min_value = th.min(h)
            print("最小值:", min_value)

            if input_sum==9:
                print("----+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("9--------------")
                print("h.shape:",h.shape)
                b, c, hh, w = h.shape
                h1=h
                h1 = h1.view(b, 512, -1).permute(0, 2, 1)

                s_fused = self.attention_module1(h1, s, s)

                v_fused = self.attention_module1(h1, v, v)

                s_fused = s_fused.permute(0, 2, 1).view(b, 512, 64, 64)
                v_fused = v_fused.permute(0, 2, 1).view(b, 512, 64, 64)







                mix_logits = self.mix1(torch.cat([s_fused, v_fused], dim=1))
                w = torch.softmax(mix_logits / self.mix_tau, dim=1)
                w_s, w_v = w[:, 0:1], w[:, 1:2]
                combined_fused_features = w_s * s_fused + w_v * v_fused

                g = torch.sigmoid(self.resgate1(h))
                h = h + g * combined_fused_features








            if input_sum==12:
                print("----+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("12-------------")


                b, c, hh, w = h.shape
                h2 = h
                h2 = h2.view(b, 512, -1).permute(0, 2, 1)

                s_fused = self.attention_module2(h2, s, s)

                v_fused = self.attention_module2(h2, v, v)

                s_fused = s_fused.permute(0, 2, 1).view(b, 512, 32, 32)
                v_fused = v_fused.permute(0, 2, 1).view(b, 512, 32, 32)







                mix_logits = self.mix2(torch.cat([s_fused, v_fused], dim=1))
                w = torch.softmax(mix_logits / self.mix_tau, dim=1)
                w_s, w_v = w[:, 0:1], w[:, 1:2]
                combined_fused_features = w_s * s_fused + w_v * v_fused

                g = torch.sigmoid(self.resgate2(h))
                h = h + g * combined_fused_features







            if input_sum==15:
                print("----+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("15-------------")
                b, c, hh, w = h.shape
                h2 = h
                h2 = self.channel_adjust(h2)
                h2 = h2.view(b, 512, -1).permute(0, 2, 1)

                s_fused = self.attention_module3(h2, s, s)

                v_fused = self.attention_module3(h2, v, v)

                s_fused = s_fused.permute(0, 2, 1).view(b, 512, 16, 16)
                v_fused = v_fused.permute(0, 2, 1).view(b, 512, 16, 16)

                s_fused = self.channel_adjust1(s_fused)
                v_fused = self.channel_adjust2(v_fused)







                mix_logits = self.mix3(torch.cat([s_fused, v_fused], dim=1))
                w = torch.softmax(mix_logits / self.mix_tau, dim=1)
                w_s, w_v = w[:, 0:1], w[:, 1:2]
                combined_fused_features = w_s * s_fused + w_v * v_fused

                g = torch.sigmoid(self.resgate3(h))
                h = h + g * combined_fused_features










            hs.append(h)

        print("----middle_block----")
        h = self.middle_block(h, emb)
        print("h.shape:", h.shape)

        print("----output_blocks----")
        output_sum = 0
        for module in self.output_blocks:
            output_sum=output_sum+1
            h = th.cat([h, hs.pop()], dim=1)
            print("shuru output before:",h.shape)
            h = module(h, emb)
            print("output_sum:", output_sum)
            print("h.shape:", h.shape)
        h = h.float()
        h1=h
        print("unet ouput.shape:",self.out(h1).shape)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))


        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

if __name__ == '__main__':
    unet=UNetModel(
        image_size=256,

        in_channels=7,
        model_channels=256,


        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=tuple([8,16,32]),
        dropout= 0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,)
    unet.to('cpu')
    x=th.randn(1, 7, 256, 256).to('cpu')
    jiazao_time1 = th.tensor([249])
    tensor1 = jiazao_time1.long().to('cpu')
    x1 = th.randn(1, 16, 16, 768).to('cpu')
    x2 = th.randn(1, 16, 16, 768).to('cpu')
    x3 = th.randn(1, 16, 16, 768).to('cpu')
    x4 = th.randn(1, 16, 16, 768).to('cpu')
    x5 = th.randn(1, 16, 16, 768).to('cpu')

    sam_encoder_feature=[x1,x2,x3,x4,x5]
    unet(x,tensor1,sam_encoder_feature)
