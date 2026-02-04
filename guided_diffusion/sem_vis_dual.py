












































































































































































































































































































































































































































































































import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualCrossAttention(nn.Module):
    """
    双分支交叉注意力（内置适配器 + 逐token动态门）：
      - 文本侧：Residual-MLP 适配 + g_txt=σ(Linear(LN(text))) 的逐token标量门
      - 图像侧：1x1 Conv(+可选 DW 3x3) 适配 + g_img=σ(Linear(LN(img))) 的逐token标量门
      - 然后投影到 d_model 做 Text->Image / Image->Text 双向 cross-attn
    """
    def __init__(self, d_txt=512, d_img=768, d_model=512, n_heads=8,
                 p_dropout=0.1, use_ln=True, mlp_ratio=4,
                 use_dwconv_on_img=True, gate_init_p=0.1):
        super().__init__()
        self.use_ln = use_ln
        self.use_dwconv_on_img = use_dwconv_on_img
        hidden_txt = int(d_txt * mlp_ratio)
        self.txt_adapt_ln   = nn.LayerNorm(d_txt)
        self.txt_adapt_fc1  = nn.Linear(d_txt, hidden_txt)
        self.txt_adapt_fc2  = nn.Linear(hidden_txt, d_txt)
        self.txt_adapt_drop = nn.Dropout(p_dropout)
        nn.init.zeros_(self.txt_adapt_fc2.weight)
        nn.init.zeros_(self.txt_adapt_fc2.bias)
        init_logit = math.log(gate_init_p) - math.log(1 - gate_init_p)
        self.txt_gate_ln = nn.LayerNorm(d_txt)
        self.txt_gate_fc = nn.Linear(d_txt, 1)
        nn.init.zeros_(self.txt_gate_fc.weight)
        nn.init.constant_(self.txt_gate_fc.bias, init_logit)

        hidden_img = int(d_img * mlp_ratio)
        self.img_adapt_ln   = nn.LayerNorm(d_img)
        self.img_conv1 = nn.Conv2d(d_img, hidden_img, kernel_size=1, bias=True)
        self.img_dwconv = nn.Conv2d(hidden_img, hidden_img, kernel_size=3, padding=1,
                                    groups=hidden_img, bias=True)
        self.img_conv2 = nn.Conv2d(hidden_img, d_img, kernel_size=1, bias=True)
        self.img_adapt_drop = nn.Dropout(p_dropout)
        nn.init.zeros_(self.img_conv2.weight)
        nn.init.zeros_(self.img_conv2.bias)
        self.img_gate_ln = nn.LayerNorm(d_img)
        self.img_gate_fc = nn.Linear(d_img, 1)
        nn.init.zeros_(self.img_gate_fc.weight)
        nn.init.constant_(self.img_gate_fc.bias, init_logit)
        self.proj_txt = nn.Linear(d_txt, d_model)
        self.proj_img = nn.Linear(d_img, d_model)
        self.xattn_t2i = nn.MultiheadAttention(d_model, n_heads, dropout=p_dropout, batch_first=True)
        self.xattn_i2t = nn.MultiheadAttention(d_model, n_heads, dropout=p_dropout, batch_first=True)
        self.ffn_txt = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(p_dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(p_dropout),
            nn.Linear(4*d_model, d_model)
        )

        if use_ln:
            self.ln_txt_q = nn.LayerNorm(d_model)
            self.ln_img_kv = nn.LayerNorm(d_model)
            self.ln_img_q  = nn.LayerNorm(d_model)
            self.ln_txt_kv = nn.LayerNorm(d_model)
            self.ln_s = nn.LayerNorm(d_model)
            self.ln_v = nn.LayerNorm(d_model)

    def forward(
        self,
        text_seq,
        img_seq,
        txt_key_padding_mask=None,
        img_key_padding_mask=None
    ):
        B, N_img, C_img = img_seq.shape

        h_txt = self.txt_adapt_ln(text_seq)
        h_txt = self.txt_adapt_fc2(F.gelu(self.txt_adapt_fc1(h_txt)))
        h_txt = self.txt_adapt_drop(h_txt)
        g_txt = torch.sigmoid(self.txt_gate_fc(self.txt_gate_ln(text_seq)))
        if txt_key_padding_mask is not None:
            g_txt = g_txt.masked_fill(txt_key_padding_mask.unsqueeze(-1), 0.0)
        text_seq = text_seq + g_txt * h_txt
        h_img = self.img_adapt_ln(img_seq)
        H = int(N_img ** 0.5)
        use_grid = self.use_dwconv_on_img and (H * H == N_img)

        if use_grid:
            h_img = h_img.permute(0, 2, 1).contiguous().view(B, C_img, H, H)
            h_img = self.img_conv1(h_img); h_img = F.gelu(h_img)
            h_img = self.img_dwconv(h_img); h_img = F.gelu(h_img)
            h_img = self.img_conv2(h_img)
            h_img = h_img.view(B, C_img, N_img).permute(0, 2, 1).contiguous()
        else:
            h_img = h_img.reshape(B * N_img, C_img, 1, 1)
            h_img = self.img_conv1(h_img); h_img = F.gelu(h_img)
            h_img = self.img_conv2(h_img)
            h_img = h_img.view(B, N_img, C_img)

        h_img = self.img_adapt_drop(h_img)
        g_img = torch.sigmoid(self.img_gate_fc(self.img_gate_ln(img_seq)))
        if img_key_padding_mask is not None:
            g_img = g_img.masked_fill(img_key_padding_mask.unsqueeze(-1), 0.0)
        img_seq = img_seq + g_img * h_img


        T = self.proj_txt(text_seq)
        I = self.proj_img(img_seq)

        T_q = self.ln_txt_q(T) if self.use_ln else T
        I_kv = self.ln_img_kv(I) if self.use_ln else I
        s, _ = self.xattn_t2i(query=T_q, key=I_kv, value=I_kv,
                              key_padding_mask=img_key_padding_mask)
        s = s + T
        s = s + (self.ffn_txt(self.ln_s(s) if self.use_ln else s))
        I_q  = self.ln_img_q(I) if self.use_ln else I
        T_kv = self.ln_txt_kv(T) if self.use_ln else T
        v, _ = self.xattn_i2t(query=I_q, key=T_kv, value=T_kv,
                              key_padding_mask=txt_key_padding_mask)
        v = v + I
        v = v + (self.ffn_img(self.ln_v(v) if self.use_ln else v))

        return s, v