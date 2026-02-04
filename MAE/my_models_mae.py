


import numpy as np






import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from MAE.util.pos_embed import get_2d_sincos_pos_embed
from MAE.vision_transformer import Block
from MAE.mae.modeling.common import LayerNorm2d, MLPBlock
from typing import Optional, Tuple, Type


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Since [cls] is useless in inpainting, we remove it.
    """

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1280,
                 depth=32,
                 num_heads=16,
                 mlp_ratio=4.,

                 out_chans=256,
                 qkv_bias= True,
                 norm_layer=nn.LayerNorm,
                 act_layer= nn.GELU,

                 use_abs_pos = True,
                 use_rel_pos = True,

                 rel_pos_zero_init = True,
                 window_size = 14,
                 global_attn_indexes=[7,15,23,31],
                 decoder_embed_dim=1024,
                 decoder_depth=26, decoder_num_heads=16,
                 norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False
                 ):

        super().__init__()




        self.sam_patch_embed = SamPatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.sam_pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:

            self.sam_pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.sam_blocks = nn.ModuleList()
        for i in range(depth):
            sam_block = Sam_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,





                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.sam_blocks.append(sam_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_decoder=mask_decoder)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)


        self.norm_pix_loss = norm_pix_loss
        self.random_mask = random_mask
        self.mask_decoder = mask_decoder

        if init:
            self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        print("pos_embed.shape:",pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        print("----unpatchify----")
        print("x.shape:",x.shape)
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        print("imgs.shape:",imgs.shape)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def adaptive_random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask: [N, 1, 256, 256]
        """
        print("----adaptive_random_masking----")
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        N, L, D = x.shape
        s = int(np.sqrt(L))
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask[mask > 0] = 1
        mask = mask.reshape(N, L)
        print("mask.shape:", mask.shape)
        len_keep = int(L * (1 - mask_ratio))
        print("len_keep:",len_keep)
        noise = torch.rand(N, L, device=x.device)

        noise = torch.clamp(noise + mask, 0.0, 1.0)
        print("noise.shape:",noise.shape)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        print("ids_shuffle.shape:",ids_shuffle.shape)
        print("ids_restore.shape:",ids_restore.shape)
        ids_keep = ids_shuffle[:, :len_keep]
        print("ids_keep.shape:",ids_keep.shape)
        print("ids_keep.unsqueeze(-1).repeat(1, 1, D).shape:",ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        print("x_masked.shape:",x_masked.shape)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print("x_masked.shape:",x_masked.shape)
        print("mask.shape:",mask.shape)
        print("ids_restore.shape:",ids_restore.shape)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask, mask_ratio):
        print("----forward_encoder----")
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        x = self.patch_embed(x)
        print("patch_embed x.shape:",x.shape)
        x = x + self.pos_embed
        if self.random_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.adaptive_random_masking(x, mask, mask_ratio)
        for blk in self.blocks:
            x, _ = blk(x)
        print("x.shape:",x.shape)
        x = self.norm(x)
        print("x.shape:", x.shape)
        print("mask.shape:",mask.shape)
        print("ids_restore.shape:",ids_restore.shape)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''
        forward decoder during training needs ids_restore
        '''
        print("----forward_decoder----")
        print("x.shape:",x.shape)
        x = self.decoder_embed(x)
        print("x.shape:",x.shape)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        print("mask_tokens.shape:",mask_tokens.shape)
        x_ = torch.cat([x, mask_tokens], dim=1)
        print("x_.shape:",x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        print("x_.shape:", x_.shape)
        x = x_
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        print("x.shape:",x.shape)
        return x
    def forward_encoder_with_mask(self, x, mask):
        print("----forward_encoder_with_mask----")
        x = self.patch_embed(x)
        print("x.shape:",x.shape)
        x = x + self.pos_embed
        N, L, D = x.shape
        s = int(np.sqrt(L))

        mask = F.interpolate(mask, size=[s, s], mode='area')
        print("mask.shape:",mask.shape)
        mask_small = mask.clone()
        mask[mask > 0] = 1
        mask_small[mask_small < 1] = 0
        mask = mask.reshape(N, L).unsqueeze(1).unsqueeze(1)
        print("mask_small.shape:",mask_small.shape)
        print("mask.shape:",mask.shape)
        for blk in self.blocks:
            x, _ = blk(x, mask)
        x = self.norm(x)
        mask = mask.squeeze(1).squeeze(1)
        mask_small = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1)
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        print("mask_small.shape:",mask_small.shape)
        return x, mask, mask_small

    def forward_decoder_with_mask(self, x, mask):
        print("----forward_decoder_with_mask----")
        x = self.decoder_embed(x)
        N, L, D = x.shape
        mask = mask.unsqueeze(-1)

        print("self.mask_token.shape:",self.mask_token.shape)
        print("self.mask_token.repeat(N, L, 1).shape:",self.mask_token.repeat(N, L, 1).shape)
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x, _ = blk(x)

        print("x.shape:",x.shape)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        print("x.shape:", x.shape)
        return x

    def forward_decoder_return_feature(self, x, mask, mask_small):
        x = self.decoder_embed(x)
        N, L, D = x.shape
        mask = mask.unsqueeze(-1)

        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask
        x = x + self.decoder_pos_embed
        scores = []
        for blk in self.decoder_blocks:
            if self.mask_decoder:
                x, score = blk(x, mask_small)
            else:
                x, score = blk(x)
            scores.append(score.unsqueeze(1))
        scores = torch.mean(torch.cat(scores, dim=1), dim=1)
        x = self.decoder_norm(x)
        return x, scores

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        print("loss.shape:",loss.shape)
        return loss

    def forward(self, imgs, mask, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        """
        print("----model forward----")
        print("imgs.shape:",imgs.shape)
        print("mask.shape:",mask.shape)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_return_feature(self, imgs, mask):
        """
        return pred(feature), scores(attention). Used during finetuning.
        """
        latent, new_mask, mask_small = self.forward_encoder_with_mask(imgs, mask)
        pred, scores = self.forward_decoder_return_feature(latent, new_mask, mask_small)
        N, L, D = pred.shape
        s = int(np.sqrt(L))
        pred = pred.reshape(N, s, s, D).permute(0, 3, 1, 2)
        return pred, scores

    def forward_return_image(self, imgs, mask):
        """
        test时：1,3,256,256   1,1,256,256
        return Image, new_mask. Used during testing.
        """
        print("----forward_return_image----")
        print("imgs.shape:",imgs.shape)
        print("mask.shape:",mask.shape)
        latent, new_mask, _ = self.forward_encoder_with_mask(imgs, mask)
        print("latent.shape:",latent.shape)
        print("new_mask.shape:",new_mask.shape)
        image = self.forward_decoder_with_mask(latent, new_mask)
        print("image.shape:",image.shape)
        image = self.unpatchify(image)
        return image, new_mask



class SamPatchEmbed(nn.Module):
    """
    Image to Patch Embedding. 本质上就是经过卷积+permute
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("----patch embedding----")
        print("x.shape:",x.shape)
        x = self.proj(x)
        print("x.shape:",x.shape)
        x = x.permute(0, 2, 3, 1)
        print("x.shape:", x.shape)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    不重叠窗口划分
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size. 比如 14

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    恢复原始中间特征尺寸
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class Sam_Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.  **这里  是不是用窗口注意力块
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAMAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,

            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)

        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]


            x, pad_hw = window_partition(x, self.window_size)


        x = self.attn(x)
        if self.window_size > 0:

            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SAMAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            print("self.rel_pos_h.shape",self.rel_pos_h.shape)
            print("self.rel_pos_w.shape",self.rel_pos_w.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)


        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)


        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
