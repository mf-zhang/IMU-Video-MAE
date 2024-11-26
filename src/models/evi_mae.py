# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath, drop_path
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from einops import rearrange
import dgl
from dgl.nn.pytorch.glob import AvgPooling
from functools import partial

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed_video(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],  patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

class Video_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Video_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Video_Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None: # attn_head_dim is None
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias: # here
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Video_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Video_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Video_DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Video_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else: # here
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None: # here
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): # num_of_patches, embed_dim
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy
    import numpy as np
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    from .gin import GIN, create_norm
    from .gat import GAT
    from .gcn import GCN
    from .dot_gat import DotGAT

    # print(f"m_type: {m_type}") # gin
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod

def sce_loss(x, y, alpha=3):

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class EVIMAE(nn.Module):
    """ EVI-MAE Model
    """
    def __init__(self, norm_pix_loss=False, tr_pos=False, video_model_dict=None, imu_model_dict=None):
        super().__init__()
        print('A EVI-MAE Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        self.pretrain_modality = video_model_dict['pretrain_modality']

        self.video_img_size = video_model_dict['img_size']
        self.video_patch_size = video_model_dict['patch_size']
        self.video_in_chans = 3
        self.video_encoder_embed_dim = video_model_dict['encoder_embed_dim']
        self.video_tubelet_size = 2
        self.video_drop_path_rate = 0.0
        self.video_encoder_depth = video_model_dict['encoder_depth']
        self.video_encoder_num_heads = video_model_dict['encoder_num_heads']
        self.video_mlp_ratio = video_model_dict['mlp_ratio']
        self.video_qkv_bias = video_model_dict['qkv_bias']
        self.video_qk_scale = None
        self.video_drop_rate = 0.0
        self.video_attn_drop_rate = 0.0
        self.video_norm_layer = nn.LayerNorm
        self.video_init_values = 0.0
        self.video_masking_ratio = video_model_dict['masking_ratio']
        self.video_decoder_embed_dim = video_model_dict['decode_embed_dim']
        self.video_decoder_num_heads = video_model_dict['decode_num_heads']

        self.imu_patch_size = imu_model_dict['patch_size']
        self.imu_channel_num = imu_model_dict['channel_num']
        self.imu_plot_height = imu_model_dict['plot_height']
        self.imu_plot_length = imu_model_dict['target_length']
        self.imu_encoder_num_heads = imu_model_dict['encoder_num_heads']
        self.imu_encoder_depth = imu_model_dict['encoder_depth']
        self.imu_encoder_embed_dim = imu_model_dict['encoder_embed_dim']
        self.imu_masking_ratio = imu_model_dict['masking_ratio']
        self.video_imu_mlp_ratio = self.video_mlp_ratio
        self.video_imu_qkv_bias = self.video_qkv_bias
        self.video_imu_qk_scale = self.video_qk_scale
        self.video_imu_norm_layer = self.video_norm_layer
        self.imu_enable_graph = imu_model_dict['enable_graph']
        self.imu_graph_masking_ratio = imu_model_dict['imu_graph_masking_ratio']

        # unified branch
        self.unified_num_heads = self.video_encoder_num_heads
        self.unified_depth = 1
        assert self.unified_depth == 1
        self.unified_embed_dim = self.video_encoder_embed_dim

        # decoder
        self.decoder_embed_dim = self.video_decoder_embed_dim
        self.decoder_depth = 8
        self.decoder_num_heads = self.video_decoder_num_heads

        # general
        self.use_which_pos_embed = 'video'
        self.use_which_masking = 'evi'

        # the encoder part ##########################################################################
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        # patch embedding
        useless = 224 # can be any number, not used
        if self.imu_enable_graph or True:
            self.patch_embed_a = PatchEmbed(useless, self.imu_patch_size, 3, self.imu_encoder_embed_dim) # 3 is xyz acceleration of each IMU
        else:
            self.patch_embed_a = PatchEmbed(useless, self.imu_patch_size, self.imu_channel_num, self.imu_encoder_embed_dim)
        self.patch_embed_video = PatchEmbed_video(img_size=self.video_img_size, patch_size=self.video_patch_size, in_chans=self.video_in_chans, embed_dim=self.video_encoder_embed_dim, tubelet_size=self.video_tubelet_size)
        self.imu_patch_width_num = int(self.imu_plot_length/self.imu_patch_size)
        self.imu_patch_height_num = int(self.imu_plot_height/self.imu_patch_size)
        self.patch_embed_a.num_patches = int(self.imu_patch_width_num * self.imu_patch_height_num)
        print('Number of imu Patches: {:d}, Video Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_video.num_patches))

        # modality embedding
        self.modality_a = nn.Parameter(torch.zeros(1, 1, self.imu_encoder_embed_dim))
        self.modality_video = nn.Parameter(torch.zeros(1, 1, self.video_encoder_embed_dim))

        # position embedding
        if self.use_which_pos_embed == 'evi':
            self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, self.imu_encoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding            
            self.pos_embed_video = nn.Parameter(torch.zeros(1, self.patch_embed_video.num_patches, self.video_encoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        elif self.use_which_pos_embed == 'video':
            self.pos_embed_a = get_sinusoid_encoding_table(self.patch_embed_a.num_patches, self.imu_encoder_embed_dim)
            self.pos_embed_video = get_sinusoid_encoding_table(self.patch_embed_video.num_patches, self.video_encoder_embed_dim)

        # imu-branch
        self.blocks_a = nn.ModuleList([
            Block(
                self.imu_encoder_embed_dim, self.imu_encoder_num_heads, self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, norm_layer=self.video_imu_norm_layer) 
            for i in range(self.imu_encoder_depth)])
        # video-branch
        dpr = [x.item() for x in torch.linspace(0, self.video_drop_path_rate, self.video_encoder_depth)]  # stochastic depth decay rule
        self.blocks_video = nn.ModuleList([
            Video_Block(
                dim=self.video_encoder_embed_dim, num_heads=self.video_encoder_num_heads, mlp_ratio=self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, drop=self.video_drop_rate, attn_drop=self.video_attn_drop_rate, drop_path=dpr[i], norm_layer=self.video_imu_norm_layer, init_values=self.video_init_values)
            for i in range(self.video_encoder_depth)])

        # unified branch
        self.blocks_u = nn.ModuleList([
            Block(
                self.unified_embed_dim, self.unified_num_heads, self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, norm_layer=self.video_imu_norm_layer) 
            for i in range(self.unified_depth)])

        # independent normalization layer for imu, visual, and imu-visual
        self.norm_a, self.norm_video, self.norm = self.video_imu_norm_layer(self.unified_embed_dim), self.video_imu_norm_layer(self.video_encoder_embed_dim), self.video_imu_norm_layer(self.unified_embed_dim)

        # the decoder part ##########################################################################
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(self.unified_embed_dim, self.decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_modality_video = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        if self.use_which_pos_embed == 'evi':
            self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, self.decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
            self.decoder_pos_embed_video = nn.Parameter(torch.zeros(1, self.patch_embed_video.num_patches, self.decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        elif self.use_which_pos_embed == 'video':
            self.decoder_pos_embed_a = get_sinusoid_encoding_table(self.patch_embed_a.num_patches, self.decoder_embed_dim)
            self.decoder_pos_embed_video = get_sinusoid_encoding_table(self.patch_embed_video.num_patches, self.decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(
                self.decoder_embed_dim, self.decoder_num_heads, self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, norm_layer=self.video_imu_norm_layer) 
            for i in range(self.decoder_depth)])

        self.decoder_norm = self.video_imu_norm_layer(self.decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(self.decoder_embed_dim, self.imu_patch_size ** 2 * self.imu_channel_num, bias=True)  # decoder to patch
        self.decoder_pred_video = nn.Linear(self.decoder_embed_dim, self.video_patch_size ** 2 * self.video_in_chans * self.video_tubelet_size, bias=True)  # decoder to patch
        self.norm_pix_loss = norm_pix_loss

        self.video_pooling = nn.AdaptiveAvgPool2d(1)

        self.initialize_weights()

        print('imu Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Video Positional Embedding Shape:', self.pos_embed_video.shape)

        if self.imu_enable_graph:
            self.graph_enc_mask_token = nn.Parameter(torch.zeros(1, self.pos_embed_a.shape[2]))

            g_num_hidden = 512
            g_encoder_type = imu_model_dict['imu_graph_net'] # 'gin' ok,  'gat' no
            g_num_layers = 2
            g_nhead = 2
            g_decoder_type = g_encoder_type
            g_nhead_out = 1
            g_in_dim = self.pos_embed_a.shape[2]
            g_activation = 'prelu'
            g_feat_drop = 0.2
            g_attn_drop = 0.1
            g_negative_slope = 0.2
            g_residual = False
            g_norm = 'batchnorm'

            assert g_num_hidden % g_nhead == 0
            assert g_num_hidden % g_nhead_out == 0
            if g_encoder_type in ("gat", "dotgat"):
                g_enc_num_hidden = g_num_hidden // g_nhead
                g_enc_nhead = g_nhead
            else:
                g_enc_num_hidden = g_num_hidden
                g_enc_nhead = 1

            g_dec_in_dim = g_num_hidden
            g_dec_num_hidden = g_num_hidden // g_nhead_out if g_decoder_type in ("gat", "dotgat") else g_num_hidden
            g_dec_n_head_out = g_nhead_out

            # build encoder
            self.graph_encoder = setup_module(
                m_type=g_encoder_type,
                enc_dec="encoding",
                in_dim=g_in_dim,
                num_hidden=g_enc_num_hidden,
                out_dim=g_enc_num_hidden,
                num_layers=g_num_layers,
                nhead=g_enc_nhead,
                nhead_out=g_enc_nhead,
                concat_out=True,
                activation=g_activation,
                dropout=g_feat_drop,
                attn_drop=g_attn_drop,
                negative_slope=g_negative_slope,
                residual=g_residual,
                norm=g_norm,
            )

            # build decoder for attribute prediction
            self.graph_decoder = setup_module(
                m_type=g_decoder_type,
                enc_dec="decoding",
                in_dim=g_dec_in_dim,
                num_hidden=g_dec_num_hidden,
                out_dim=g_in_dim,
                num_layers=1,
                nhead=g_nhead,
                nhead_out=g_dec_n_head_out,
                activation=g_activation,
                dropout=g_feat_drop,
                attn_drop=g_attn_drop,
                negative_slope=g_negative_slope,
                residual=g_residual,
                norm=g_norm,
                concat_out=True,
            )

            self.graph_encoder_to_decoder = nn.Linear(g_dec_in_dim, g_dec_in_dim, bias=False)

            loss_fn = 'sce'
            alpha_l = 1
            self.g_criterion = self.g_setup_loss_fn(loss_fn, alpha_l)

    def g_setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        
        if self.use_which_pos_embed == 'evi':
            embed_dim_here = self.pos_embed_a.shape[-1]
            assert embed_dim_here == self.imu_encoder_embed_dim
            pos_embed_a = get_2d_sincos_pos_embed(embed_dim_here, self.imu_patch_height_num, self.imu_patch_width_num, cls_token=False)
            self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
            
            assert int(self.patch_embed_video.num_patches ** .5) == self.patch_embed_video.num_patches ** .5
            pos_embed_video = get_2d_sincos_pos_embed(self.video_encoder_embed_dim, int(self.patch_embed_video.num_patches ** .5), int(self.patch_embed_video.num_patches ** .5), cls_token=False)
            self.pos_embed_video.data.copy_(torch.from_numpy(pos_embed_video).float().unsqueeze(0))
            
            decoder_embed_dim_here = self.decoder_pos_embed_a.shape[-1]
            assert decoder_embed_dim_here == self.decoder_embed_dim
            decoder_pos_embed_a = get_2d_sincos_pos_embed(decoder_embed_dim_here, self.imu_patch_height_num, self.imu_patch_width_num, cls_token=False)
            self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))
            
            decoder_pos_embed_video = get_2d_sincos_pos_embed(self.decoder_embed_dim, int(self.patch_embed_video.num_patches ** .5), int(self.patch_embed_video.num_patches ** .5), cls_token=False)
            self.decoder_pos_embed_video.data.copy_(torch.from_numpy(decoder_pos_embed_video).float().unsqueeze(0))
        elif self.use_which_pos_embed == 'video':
            pass

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_video.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_video, std=.02)

        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_video, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, c, h, w, p=16, modality=None, video_frame_num=None):
        
        if modality == 'a':
            """
            imgs: (B, c, h_pixels, w_pixels)
            patch_num = h*w = (h_pixels/p) * (w_pixels/p)
            x: (B, patch_num, patch_size**2 *c)
            """
            x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
            return x
        elif modality == 'video':
            """
            imgs: (B, 3, video_frame_num, h_pixels, w_pixels)
            patch_num = video_frame_num * h * w = video_frame_num * (h_pixels/p) * (w_pixels/p)
            x = (B, patch_num, patch_size**2 *3)
            """
            print('should not reach here')
            exit()
            print(imgs.shape)
            x = imgs.reshape(shape=(imgs.shape[0], video_frame_num, c, h, p, w, p))

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        t = 20
        f = 8

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the imu patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, a, v, generated_video_mask, mask_ratio_a, mask_ratio_v, mask_mode='unstructured'):
        if self.pretrain_modality == 'both':
            assert len(a.shape) == 4
            assert a.shape[1] == self.imu_channel_num

            a_left_arm = a[:, 0:3, :, :]
            a_right_arm = a[:, 3:6, :, :]
            a_left_leg = a[:, 6:9, :, :]
            a_right_leg = a[:, 9:12, :, :]
            a = torch.cat((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0) # b*4 3 320 128

            bs = int(a.shape[0]/4)

            # IMU patchify
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
            a = a + self.modality_a

            v = self.patch_embed_video(v)
            v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
            v = v + self.modality_video

            ############ Graph MAE ############
            if self.imu_enable_graph:
                # STEP 1: graph construction
                if True:
                    a_for_graph = a.clone()
                    for blk in self.blocks_a:
                        a_for_graph = blk(a_for_graph)
                    a_left_arm_for_graph = a_for_graph[0:bs, :, :]
                    a_right_arm_for_graph = a_for_graph[bs:2*bs, :, :]
                    a_left_leg_for_graph = a_for_graph[2*bs:3*bs, :, :]
                    a_right_leg_for_graph = a_for_graph[3*bs:4*bs, :, :]
                    a_left_arm_for_graph = torch.mean(a_left_arm_for_graph, dim=1)
                    a_right_arm_for_graph = torch.mean(a_right_arm_for_graph, dim=1)
                    a_left_leg_for_graph = torch.mean(a_left_leg_for_graph, dim=1)
                    a_right_leg_for_graph = torch.mean(a_right_leg_for_graph, dim=1)

                    u_edge, v_edge = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]), torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
                    body_graph = dgl.graph((u_edge, v_edge))
                    body_graph = body_graph.to(a.device)
                    body_graphs = []
                    for bi in range(bs):
                        body_graph_i = body_graph.clone()
                        stacked_features = torch.stack((a_left_arm_for_graph[bi], a_right_arm_for_graph[bi], a_left_leg_for_graph[bi], a_right_leg_for_graph[bi]), dim=0)
                        body_graph_i.ndata['attr'] = stacked_features
                        body_graphs.append(body_graph_i)

                    body_graphs_batch = dgl.batch(body_graphs)
                    body_graphs_batch_feat = body_graphs_batch.ndata["attr"]

                # STEP 2: encoding_mask_noise
                if True:
                    mask_rate = self.imu_graph_masking_ratio
                    num_nodes = body_graphs_batch.num_nodes()
                    perm = torch.randperm(num_nodes, device=body_graphs_batch_feat.device)
                    num_mask_nodes = int(mask_rate * num_nodes)

                    num_mask_nodes = int(mask_rate * num_nodes)
                    mask_nodes = perm[: num_mask_nodes]
                    keep_nodes = perm[num_mask_nodes: ]

                    body_graphs_batch_feat_masked = body_graphs_batch_feat.clone()
                    body_graphs_batch_feat_masked[mask_nodes] = 0.0

                    body_graphs_batch_feat_masked[mask_nodes] += self.graph_enc_mask_token
                    body_graphs_batch_use = body_graphs_batch.clone()

                # STEP 3: graph encoding
                if True:
                    enc_rep, all_hidden = self.graph_encoder(body_graphs_batch_use, body_graphs_batch_feat_masked, return_hidden=True)

                # STEP 4: feature reconstruction
                if True:
                    rep = self.graph_encoder_to_decoder(enc_rep)
                    rep[mask_nodes] = 0
                    recon = self.graph_decoder(body_graphs_batch_use, rep)

                # STEP 5: loss
                if True:
                    body_graphs_batch_feat_init = body_graphs_batch_feat[mask_nodes]
                    body_graphs_batch_feat_rec = recon[mask_nodes]
                    g_loss = self.g_criterion(body_graphs_batch_feat_rec, body_graphs_batch_feat_init)

            ###################################

            ############ EVI MAE ##############

            a_left_arm = a[0:bs, :, :]
            a_right_arm = a[bs:2*bs, :, :]
            a_left_leg = a[2*bs:3*bs, :, :]
            a_right_leg = a[3*bs:4*bs, :, :]
            a_mean = torch.mean(torch.stack((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0), dim=0)
            a = a_mean

            # by default, we always use unstructured masking
            if mask_mode == 'unstructured':
                a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
            else:
                a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

            # visual branch always use unstructured masking
            v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)
                
            # imu and visual stream, independent blocks
            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_video:
                v = blk(v)
                
            if a.shape[2] != v.shape[2]:
                if a.shape[2] == 768 and v.shape[2] == 384:
                    a = F.avg_pool1d(a, kernel_size=2, stride=2)
                else:
                    print('not implemented yet')
                    exit()

            x = torch.cat((a, v), dim=1)

            # unified stream, shared blocks_u, but independent normalization layers
            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            for blk in self.blocks_u:
                ca = blk(a, 'a')
            ca = self.norm_a(ca)

            for blk in self.blocks_u:
                cv = blk(v, 'v')
            cv = self.norm_video(cv)

            if self.imu_enable_graph:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, g_loss
            else:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, torch.tensor(0).to(a.device)
        if self.pretrain_modality == 'imu':

            assert len(a.shape) == 4
            assert a.shape[1] == self.imu_channel_num

            a_left_arm = a[:, 0:3, :, :]
            a_right_arm = a[:, 3:6, :, :]
            a_left_leg = a[:, 6:9, :, :]
            a_right_leg = a[:, 9:12, :, :]
            a = torch.cat((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0)

            bs = int(a.shape[0]/4)


            # IMU patchify
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
            a = a + self.modality_a

            ############ Graph MAE ############
            if self.imu_enable_graph:
                # STEP 1: graph construction
                if True:
                    a_for_graph = a.clone()
                    for blk in self.blocks_a:
                        a_for_graph = blk(a_for_graph)
                    a_left_arm_for_graph = a_for_graph[0:bs, :, :]
                    a_right_arm_for_graph = a_for_graph[bs:2*bs, :, :]
                    a_left_leg_for_graph = a_for_graph[2*bs:3*bs, :, :]
                    a_right_leg_for_graph = a_for_graph[3*bs:4*bs, :, :]
                    a_left_arm_for_graph = torch.mean(a_left_arm_for_graph, dim=1)
                    a_right_arm_for_graph = torch.mean(a_right_arm_for_graph, dim=1)
                    a_left_leg_for_graph = torch.mean(a_left_leg_for_graph, dim=1)
                    a_right_leg_for_graph = torch.mean(a_right_leg_for_graph, dim=1)

                    u_edge, v_edge = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]), torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
                    body_graph = dgl.graph((u_edge, v_edge))
                    body_graph = body_graph.to(a.device)
                    body_graphs = []
                    for bi in range(bs):
                        body_graph_i = body_graph.clone()
                        stacked_features = torch.stack((a_left_arm_for_graph[bi], a_right_arm_for_graph[bi], a_left_leg_for_graph[bi], a_right_leg_for_graph[bi]), dim=0)
                        body_graph_i.ndata['attr'] = stacked_features
                        body_graphs.append(body_graph_i)

                    body_graphs_batch = dgl.batch(body_graphs)
                    body_graphs_batch_feat = body_graphs_batch.ndata["attr"]

                # STEP 2: encoding_mask_noise
                if True:
                    mask_rate = self.imu_graph_masking_ratio
                    num_nodes = body_graphs_batch.num_nodes()
                    perm = torch.randperm(num_nodes, device=body_graphs_batch_feat.device)
                    num_mask_nodes = int(mask_rate * num_nodes)

                    # random masking
                    num_mask_nodes = int(mask_rate * num_nodes)
                    mask_nodes = perm[: num_mask_nodes]
                    keep_nodes = perm[num_mask_nodes: ]

                    body_graphs_batch_feat_masked = body_graphs_batch_feat.clone()
                    body_graphs_batch_feat_masked[mask_nodes] = 0.0

                    body_graphs_batch_feat_masked[mask_nodes] += self.graph_enc_mask_token
                    body_graphs_batch_use = body_graphs_batch.clone()

                # STEP 3: graph encoding
                if True:
                    enc_rep, all_hidden = self.graph_encoder(body_graphs_batch_use, body_graphs_batch_feat_masked, return_hidden=True)

                # STEP 4: feature reconstruction
                if True:
                    rep = self.graph_encoder_to_decoder(enc_rep)
                    rep[mask_nodes] = 0
                    recon = self.graph_decoder(body_graphs_batch_use, rep)

                # STEP 5: loss
                if True:
                    body_graphs_batch_feat_init = body_graphs_batch_feat[mask_nodes]
                    body_graphs_batch_feat_rec = recon[mask_nodes]

                    g_loss = self.g_criterion(body_graphs_batch_feat_rec, body_graphs_batch_feat_init)

            ###################################

            ############ EVI MAE ##############

            a_left_arm = a[0:bs, :, :]
            a_right_arm = a[bs:2*bs, :, :]
            a_left_leg = a[2*bs:3*bs, :, :]
            a_right_leg = a[3*bs:4*bs, :, :]
            a_mean = torch.mean(torch.stack((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0), dim=0)
            a = a_mean

            # by default, we always use unstructured masking
            if mask_mode == 'unstructured':
                a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
            else:
                a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)
                
            # imu and visual stream, independent blocks
            for blk in self.blocks_a:
                a = blk(a)

            if a.shape[2] == 768:
                a = F.avg_pool1d(a, kernel_size=2, stride=2)
            else:
                print('not implemented yet')
                exit()

            x = a.clone()

            for blk in self.blocks_u:
                ca = blk(a, 'a')
            ca = self.norm_a(ca)

            mask_v = None
            ids_restore_v = None
            cv = None

            if self.imu_enable_graph:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, g_loss
            else:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, torch.tensor(0).to(a.device)
        if self.pretrain_modality == 'video':
            v = self.patch_embed_video(v) # b 3 16 224 224 -> b 1568 384
            v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach() # b 1568 384
            v = v + self.modality_video # b 1 384 - b 1568 384

            ############ EVI MAE ##############

            # visual branch always use unstructured masking
            v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)

            for blk in self.blocks_video:
                v = blk(v)
            
            x = v.clone()

            for blk in self.blocks_u:
                cv = blk(v, 'v')
            cv = self.norm_video(cv)

            mask_a = None
            ids_restore_a = None
            ca = None
            g_loss = None

            if self.imu_enable_graph:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, g_loss
            else:
                return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, torch.tensor(0).to(a.device)

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):
        if self.pretrain_modality == 'both':
            x = self.decoder_embed(x)

            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
            a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)
            a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            # similar for the visual modality
            mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
            v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)
            v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            # concatenate imu and visual tokens
            x = torch.cat([a_, v_], dim=1)

            decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_video], dim=1)
            x = x + decoder_pos_embed.type_as(x).to(x.device).clone().detach()

            # add modality indication tokens
            x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_video

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])
            x_v = self.decoder_pred_video(x[:, self.patch_embed_a.num_patches:, :])

            return x_a, x_v
        if self.pretrain_modality == 'imu':
            x = self.decoder_embed(x)

            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1) # (b, 120=160-40, 1) -> (b, 120, 192)
            a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # -> b 160=120+40 192
            a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> b 160 192

            x = a_

            decoder_pos_embed = self.decoder_pos_embed_a
            x = x + decoder_pos_embed.type_as(x).to(x.device).clone().detach() # b 1728=1568+160 192

            x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])
            x_v = None

            # return imu and video tokens
            return x_a, x_v
        if self.pretrain_modality == 'video':
            x = self.decoder_embed(x)

            # similar for the visual modality
            mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1) # (b, 1412=1568-156, 1) -> (b, 1412, 192)
            v_ = torch.cat([x[:, :, :], mask_tokens_v], dim=1)  # -> b 1568=1412+156 192
            v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> b 1568 192

            x = v_

            # decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_video], dim=1)
            decoder_pos_embed = self.decoder_pos_embed_video
            x = x + decoder_pos_embed.type_as(x).to(x.device).clone().detach() # b 1728=1568+160 192

            # add modality indication tokens
            x[:, :, :] = x[:, :, :] + self.decoder_modality_video

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x_v = self.decoder_pred_video(x[:, :, :])

            x_a = None

            return x_a, x_v

    def forward_contrastive(self, imu_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-imu representation

        assert imu_rep.shape[0] == video_rep.shape[0]
        assert imu_rep.shape[1] == video_rep.shape[1]
        assert len(imu_rep.shape) == 2
        assert len(video_rep.shape) == 2

        imu_rep = torch.nn.functional.normalize(imu_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(imu_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=imu_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=imu_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=imu_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # for imu, need to adjust the shape
            if len(input.shape) == 3: input = input.unsqueeze(1)
            elif len(input.shape) == 4: pass
            else: assert False
            input = input.transpose(2, 3)
            target = self.patchify(input, self.imu_channel_num, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), self.imu_patch_size, 'a')
        elif modality == 'v':
            patch_size_video = 16
            target = rearrange(input, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size_video, p2=patch_size_video)
            

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imu_input, video_input, v_masks, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
        mask_ratio_a = self.imu_masking_ratio
        mask_ratio_v = self.video_masking_ratio
        
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v, loss_g = self.forward_encoder(imu_input, video_input, v_masks, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        # if mae loss is used
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)
            if self.pretrain_modality == 'both':
                loss_mae_a = self.forward_mae_loss(imu_input, pred_a, mask_a, 'a')
                loss_mae_v = self.forward_mae_loss(video_input, pred_v, mask_v, 'v')
                loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
            if self.pretrain_modality == 'imu':
                loss_mae_a = self.forward_mae_loss(imu_input, pred_a, mask_a, 'a')
                loss_mae_v = torch.tensor(0.0, device=imu_input.device)
                loss_mae = mae_loss_weight * loss_mae_a
            if self.pretrain_modality == 'video':
                loss_mae_a = torch.tensor(0.0, device=imu_input.device)
                loss_mae_v = self.forward_mae_loss(video_input, pred_v, mask_v, 'v')
                loss_mae = mae_loss_weight * loss_mae_v
        else:
            loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=imu_input.device), torch.tensor(0.0, device=imu_input.device), torch.tensor(0.0, device=imu_input.device)

        # if contrastive loss is used
        if contrast_loss_weight != 0:
            # note this is single directional
            if self.pretrain_modality == 'both':
                loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1))
            if self.pretrain_modality == 'imu':
                loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_a.mean(dim=1))
            if self.pretrain_modality == 'video':
                loss_c, c_acc = self.forward_contrastive(latent_c_v.mean(dim=1), latent_c_v.mean(dim=1))
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=imu_input.device), torch.tensor(0.0, device=imu_input.device)

        loss = loss_mae + loss_c
        if self.imu_enable_graph:
            loss += loss_g * 10

        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g

# finetune EVI-MAE model
class EVIMAEFT(nn.Module):
    def __init__(self, label_dim, img_size=224, imu_length=1024, patch_size=16, in_chans=3, video_model_dict=None, imu_model_dict=None,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
        super().__init__()

        self.video_img_size = video_model_dict['img_size']
        self.video_patch_size = video_model_dict['patch_size']
        self.video_in_chans = 3
        self.video_encoder_embed_dim = video_model_dict['encoder_embed_dim']
        self.video_tubelet_size = 2
        self.video_drop_path_rate = 0.0
        self.video_encoder_depth = video_model_dict['encoder_depth']
        self.video_encoder_num_heads = video_model_dict['encoder_num_heads']
        self.video_mlp_ratio = video_model_dict['mlp_ratio']
        self.video_qkv_bias = video_model_dict['qkv_bias']
        self.video_qk_scale = None
        self.video_drop_rate = 0.0
        self.video_attn_drop_rate = 0.0
        self.video_norm_layer = nn.LayerNorm
        self.video_init_values = 0.0
        self.video_decoder_embed_dim = video_model_dict['decode_embed_dim']
        self.video_decoder_num_heads = video_model_dict['decode_num_heads']

        self.imu_patch_size = imu_model_dict['patch_size']
        self.imu_channel_num = imu_model_dict['channel_num']
        self.imu_plot_height = imu_model_dict['plot_height']
        self.imu_plot_length = imu_model_dict['target_length']
        self.imu_encoder_num_heads = imu_model_dict['encoder_num_heads']
        self.imu_encoder_depth = imu_model_dict['encoder_depth']
        self.imu_encoder_embed_dim = imu_model_dict['encoder_embed_dim']
        self.video_imu_mlp_ratio = self.video_mlp_ratio
        self.video_imu_qkv_bias = self.video_qkv_bias
        self.video_imu_qk_scale = self.video_qk_scale
        self.video_imu_norm_layer = self.video_norm_layer
        self.imu_enable_graph = imu_model_dict['enable_graph']
        self.imu_two_stream = imu_model_dict['imu_two_stream']


        # unified branch
        self.unified_num_heads = self.video_encoder_num_heads
        self.unified_depth = 1
        assert self.unified_depth == 1
        self.unified_embed_dim = self.video_encoder_embed_dim

        # general
        self.use_which_pos_embed = 'video' 
        self.use_which_masking = 'evi'

        # the encoder part ##########################################################################
        # overide the timm package
        print('Use norm_pix_loss: ', norm_pix_loss)
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        # patch embedding
        useless = 224 # can be any number, not used
        self.patch_embed_a = PatchEmbed(useless, self.imu_patch_size, 3, self.imu_encoder_embed_dim)
        self.patch_embed_video = PatchEmbed_video(img_size=self.video_img_size, patch_size=self.video_patch_size, in_chans=self.video_in_chans, embed_dim=self.video_encoder_embed_dim, tubelet_size=self.video_tubelet_size)
        self.imu_patch_width_num = int(self.imu_plot_length/self.imu_patch_size)
        self.imu_patch_height_num = int(self.imu_plot_height/self.imu_patch_size)
        self.patch_embed_a.num_patches = int(self.imu_patch_width_num * self.imu_patch_height_num)
        print('Number of imu Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_video.num_patches))

        # modality embedding
        self.modality_a = nn.Parameter(torch.zeros(1, 1, self.imu_encoder_embed_dim))
        self.modality_video = nn.Parameter(torch.zeros(1, 1, self.video_encoder_embed_dim))

        # position embedding
        if self.use_which_pos_embed == 'evi':
            self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, self.imu_encoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
            self.pos_embed_video = nn.Parameter(torch.zeros(1, self.patch_embed_video.num_patches, self.video_encoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        elif self.use_which_pos_embed == 'video':
            self.pos_embed_a = get_sinusoid_encoding_table(self.patch_embed_a.num_patches, self.imu_encoder_embed_dim)
            self.pos_embed_video = get_sinusoid_encoding_table(self.patch_embed_video.num_patches, self.video_encoder_embed_dim)

        # imu-branch
        self.blocks_a = nn.ModuleList([
            Block(
                self.imu_encoder_embed_dim, self.imu_encoder_num_heads, self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, norm_layer=self.video_imu_norm_layer) 
            for i in range(self.imu_encoder_depth)])
        
        # video-branch
        dpr = [x.item() for x in torch.linspace(0, self.video_drop_path_rate, self.video_encoder_depth)]  # stochastic depth decay rule
        self.blocks_video = nn.ModuleList([
            Video_Block(
                dim=self.video_encoder_embed_dim, num_heads=self.video_encoder_num_heads, mlp_ratio=self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, drop=self.video_drop_rate, attn_drop=self.video_attn_drop_rate, drop_path=dpr[i], norm_layer=self.video_imu_norm_layer, init_values=self.video_init_values)
            for i in range(self.video_encoder_depth)])

        # unified branch
        self.blocks_u = nn.ModuleList([
            Block(
                self.unified_embed_dim, self.unified_num_heads, self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale, norm_layer=self.video_imu_norm_layer) 
            for i in range(self.unified_depth)])

        # independent normalization layer for imu, visual, and imu-visual
        self.norm_a, self.norm_video, self.norm = self.video_imu_norm_layer(self.unified_embed_dim), self.video_imu_norm_layer(self.video_encoder_embed_dim), self.video_imu_norm_layer(self.unified_embed_dim)

        if not self.imu_enable_graph:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.unified_embed_dim), nn.Linear(self.unified_embed_dim, label_dim))
        elif self.imu_enable_graph and (not self.imu_two_stream):
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.unified_embed_dim+512), nn.Linear(self.unified_embed_dim+512, label_dim))
        elif self.imu_enable_graph and self.imu_two_stream:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.unified_embed_dim), nn.Linear(self.unified_embed_dim, label_dim))
            self.another_mlp_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, label_dim))

        self.initialize_weights()

        print('imu Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_video.shape)

        if self.imu_enable_graph:
            g_num_hidden = 512
            g_encoder_type = imu_model_dict['imu_graph_net']
            g_num_layers = 2
            g_nhead = 2
            g_decoder_type = g_encoder_type
            g_nhead_out = 1
            g_in_dim = self.pos_embed_a.shape[2]
            g_activation = 'prelu'
            g_feat_drop = 0.2
            g_attn_drop = 0.1
            g_negative_slope = 0.2
            g_residual = False
            g_norm = 'batchnorm'

            assert g_num_hidden % g_nhead == 0
            assert g_num_hidden % g_nhead_out == 0
            if g_encoder_type in ("gat", "dotgat"):
                g_enc_num_hidden = g_num_hidden // g_nhead
                g_enc_nhead = g_nhead
            else:
                g_enc_num_hidden = g_num_hidden
                g_enc_nhead = 1

            g_dec_in_dim = g_num_hidden
            g_dec_num_hidden = g_num_hidden // g_nhead_out if g_decoder_type in ("gat", "dotgat") else g_num_hidden
            g_dec_n_head_out = g_nhead_out

            # build encoder
            self.graph_encoder = setup_module(
                m_type=g_encoder_type,
                enc_dec="encoding",
                in_dim=g_in_dim,
                num_hidden=g_enc_num_hidden,
                out_dim=g_enc_num_hidden,
                num_layers=g_num_layers,
                nhead=g_enc_nhead,
                nhead_out=g_enc_nhead,
                concat_out=True,
                activation=g_activation,
                dropout=g_feat_drop,
                attn_drop=g_attn_drop,
                negative_slope=g_negative_slope,
                residual=g_residual,
                norm=g_norm,
            )

            self.graph_pooler = AvgPooling()

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        print('not implemented yet')
        exit()

    def initialize_weights(self):

        if self.use_which_pos_embed == 'evi':
            embed_dim_here = self.pos_embed_a.shape[-1]
            assert embed_dim_here == self.imu_encoder_embed_dim
            pos_embed_a = get_2d_sincos_pos_embed(embed_dim_here, self.imu_patch_height_num, self.imu_patch_width_num, cls_token=False)
            self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
            
            assert int(self.patch_embed_video.num_patches ** .5) == self.patch_embed_video.num_patches ** .5
            pos_embed_video = get_2d_sincos_pos_embed(self.video_encoder_embed_dim, int(self.patch_embed_video.num_patches ** .5), int(self.patch_embed_video.num_patches ** .5), cls_token=False)
            self.pos_embed_video.data.copy_(torch.from_numpy(pos_embed_video).float().unsqueeze(0))
            
            decoder_embed_dim_here = self.decoder_pos_embed_a.shape[-1]
            assert decoder_embed_dim_here == self.decoder_embed_dim
            decoder_pos_embed_a = get_2d_sincos_pos_embed(decoder_embed_dim_here, self.imu_patch_height_num, self.imu_patch_width_num, cls_token=False)
            self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))
            
            decoder_pos_embed_video = get_2d_sincos_pos_embed(self.decoder_embed_dim, int(self.patch_embed_video.num_patches ** .5), int(self.patch_embed_video.num_patches ** .5), cls_token=False)
            self.decoder_pos_embed_video.data.copy_(torch.from_numpy(decoder_pos_embed_video).float().unsqueeze(0))
        elif self.use_which_pos_embed == 'video':
            pass

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_video.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_video, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        if mode == 'multimodal':
            assert len(a.shape) == 4
            assert a.shape[1] == self.imu_channel_num

            # reshape a to b*4, 3, 320, 128
            a_left_arm = a[:, 0:3, :, :]
            a_right_arm = a[:, 3:6, :, :]
            a_left_leg = a[:, 6:9, :, :]
            a_right_leg = a[:, 9:12, :, :]

            a = torch.cat((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0)
            bs = int(a.shape[0]/4)

            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
            a = a + self.modality_a

            v = self.patch_embed_video(v)
            v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
            v = v + self.modality_video

            ################### Graph ###################

            if self.imu_enable_graph:
                # STEP 1: graph construction
                if True:
                    a_for_graph = a.clone()
                    for blk in self.blocks_a:
                        a_for_graph = blk(a_for_graph)
                    
                    a_left_arm_for_graph = a_for_graph[0:bs, :, :]
                    a_right_arm_for_graph = a_for_graph[bs:2*bs, :, :]
                    a_left_leg_for_graph = a_for_graph[2*bs:3*bs, :, :]
                    a_right_leg_for_graph = a_for_graph[3*bs:4*bs, :, :]
                    a_left_arm_for_graph = torch.mean(a_left_arm_for_graph, dim=1)
                    a_right_arm_for_graph = torch.mean(a_right_arm_for_graph, dim=1)
                    a_left_leg_for_graph = torch.mean(a_left_leg_for_graph, dim=1)
                    a_right_leg_for_graph = torch.mean(a_right_leg_for_graph, dim=1)
                    
                    u_edge, v_edge = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]), torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
                    body_graph = dgl.graph((u_edge, v_edge))
                    body_graph = body_graph.to(a.device)
                    body_graphs = []
                    for bi in range(bs):
                        body_graph_i = body_graph.clone()
                        stacked_features = torch.stack((a_left_arm_for_graph[bi], a_right_arm_for_graph[bi], a_left_leg_for_graph[bi], a_right_leg_for_graph[bi]), dim=0)
                        body_graph_i.ndata['attr'] = stacked_features
                        body_graphs.append(body_graph_i)

                    body_graphs_batch = dgl.batch(body_graphs)
                    body_graphs_batch_feat = body_graphs_batch.ndata["attr"]
                
                # STEP 3: graph encoding
                if True:
                    enc_rep, all_hidden = self.graph_encoder(body_graphs_batch, body_graphs_batch_feat, return_hidden=True)
                    graph_enc_rep_Bx512 = self.graph_pooler(body_graphs_batch, enc_rep)

            #############################################
                    
            ############### EVI MAE #####################
                    
            a_left_arm = a[0:bs, :, :]
            a_right_arm = a[bs:2*bs, :, :]
            a_left_leg = a[2*bs:3*bs, :, :]
            a_right_leg = a[3*bs:4*bs, :, :]
            a_mean = torch.mean(torch.stack((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0), dim=0)
            a = a_mean

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_video:
                v = blk(v)

            if a.shape[2] != v.shape[2]:
                if a.shape[2] == 768 and v.shape[2] == 384:
                    a = F.avg_pool1d(a, kernel_size=2, stride=2)
                else:
                    print('not implemented yet')
                    exit()

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            x = x.mean(dim=1)
            if self.imu_enable_graph and (not self.imu_two_stream):
                x = torch.cat((x, graph_enc_rep_Bx512), dim=1)
            elif self.imu_enable_graph and self.imu_two_stream:
                graph_pred = self.another_mlp_head(graph_enc_rep_Bx512)

            x = self.mlp_head(x)

            if self.imu_enable_graph and self.imu_two_stream:
                return x, graph_pred

            return x

        elif mode == 'ft_imuonly':
            assert len(a.shape) == 4
            assert a.shape[1] == self.imu_channel_num

            a_left_arm = a[:, 0:3, :, :]
            a_right_arm = a[:, 3:6, :, :]
            a_left_leg = a[:, 6:9, :, :]
            a_right_leg = a[:, 9:12, :, :]

            a = torch.cat((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0)
            bs = int(a.shape[0]/4)

            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
            a = a + self.modality_a

            if self.imu_enable_graph:
                # STEP 1: graph construction
                if True:
                    a_for_graph = a.clone()
                    for blk in self.blocks_a:
                        a_for_graph = blk(a_for_graph)
                    
                    a_left_arm_for_graph = a_for_graph[0:bs, :, :]
                    a_right_arm_for_graph = a_for_graph[bs:2*bs, :, :]
                    a_left_leg_for_graph = a_for_graph[2*bs:3*bs, :, :]
                    a_right_leg_for_graph = a_for_graph[3*bs:4*bs, :, :]
                    a_left_arm_for_graph = torch.mean(a_left_arm_for_graph, dim=1)
                    a_right_arm_for_graph = torch.mean(a_right_arm_for_graph, dim=1)
                    a_left_leg_for_graph = torch.mean(a_left_leg_for_graph, dim=1)
                    a_right_leg_for_graph = torch.mean(a_right_leg_for_graph, dim=1)
                        
                    u_edge, v_edge = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]), torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
                    body_graph = dgl.graph((u_edge, v_edge))
                    body_graph = body_graph.to(a.device)
                    body_graphs = []
                    for bi in range(bs):
                        body_graph_i = body_graph.clone()
                        stacked_features = torch.stack((a_left_arm_for_graph[bi], a_right_arm_for_graph[bi], a_left_leg_for_graph[bi], a_right_leg_for_graph[bi]), dim=0)
                        body_graph_i.ndata['attr'] = stacked_features
                        body_graphs.append(body_graph_i)

                    body_graphs_batch = dgl.batch(body_graphs)
                    body_graphs_batch_feat = body_graphs_batch.ndata["attr"]

                # STEP 3: graph encoding
                if True:
                    enc_rep, all_hidden = self.graph_encoder(body_graphs_batch, body_graphs_batch_feat, return_hidden=True)
                    graph_enc_rep_Bx512 = self.graph_pooler(body_graphs_batch, enc_rep)

            a_left_arm = a[0:bs, :, :]
            a_right_arm = a[bs:2*bs, :, :]
            a_left_leg = a[2*bs:3*bs, :, :]
            a_right_leg = a[3*bs:4*bs, :, :]
            a_mean = torch.mean(torch.stack((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0), dim=0)
            a = a_mean

            for blk in self.blocks_a:
                a = blk(a)

            if a.shape[2] != self.video_encoder_embed_dim:
                if a.shape[2] == 768 and self.video_encoder_embed_dim == 384:
                    a = F.avg_pool1d(a, kernel_size=2, stride=2)
                else:
                    print('not implemented yet')
                    exit()

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            x = a.mean(dim=1)

            if self.imu_enable_graph and (not self.imu_two_stream):
                x = torch.cat((x, graph_enc_rep_Bx512), dim=1)
            elif self.imu_enable_graph and self.imu_two_stream:
                graph_pred = self.another_mlp_head(graph_enc_rep_Bx512)

            x = self.mlp_head(x)

            if self.imu_enable_graph and self.imu_two_stream:
                return x, graph_pred
            
            return x

        elif mode == 'ft_videoonly':
            v = self.patch_embed_video(v)
            v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
            v = v + self.modality_video

            for blk in self.blocks_video:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_video(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)

            return x

        elif mode == 'inf_imuonly':
            if len(a.shape) == 3: a = a.unsqueeze(1)
            elif len(a.shape) == 4: pass
            else: assert False
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            if a.shape[2] != self.video_encoder_embed_dim:
                if a.shape[2] == 768 and self.video_encoder_embed_dim == 384:
                    a = F.avg_pool1d(a, kernel_size=2, stride=2)
                else:
                    print('not implemented yet')
                    exit()

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        elif mode == 'inf_videoonly':
            v = self.patch_embed_video(v)
            v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
            v = v + self.modality_video

            for blk in self.blocks_video:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_video(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)

            return x