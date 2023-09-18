import numpy as np
import math
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import normalize
import cv2


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class feature_encoder(nn.Module):
    def __init__(self, input_channel, feature_channel=np.array([64, 64, 128, 128, 256], dtype=np.int)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, feature_channel[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel[0], feature_channel[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel[1], feature_channel[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel[2], feature_channel[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel[3], feature_channel[4], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class hierarchy_encoder(nn.Module):
    def __init__(self, input_channel=256, hierarchy_group=np.array([1, 2, 4, 8, 1], dtype=np.int)):
        super(hierarchy_encoder, self).__init__()
        self.group = hierarchy_group
        self.layers = nn.ModuleList([
            nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1, groups=self.group[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=self.group[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=self.group[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=self.group[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=self.group[4]),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        b, c, h, w = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and i != 0:
                g = self.group[i // 2]
                x0, out0 = x.view(b, g, -1, h, w), out.view(b, g, -1, h, w)
                out = torch.cat([x0, out0], 2).view(b, -1, h, w)
            out = layer(out)
        return out


class decoder(nn.Module):
    def __init__(self, input_channel=128, feature_channel=np.array([128, 64, 64, 3], dtype=np.int)):
        super().__init__()
        self.decoder = nn.Sequential(
            deconv(input_channel, feature_channel[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(feature_channel[0], feature_channel[1], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(feature_channel[1], feature_channel[2], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel[2], feature_channel[3], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class features_to_embeddings(nn.Module):
    def __init__(self, patch_size=np.array([3, 7], np.int)):
        super().__init__()
        self.spa_size = patch_size[0]
        self.tem_size = patch_size[1]
        self.lin_spa = nn.Linear(1152, 1024)
        self.lin_tem = nn.Linear(6272, 1024)

    def forward(self, e):
        b, t, c, h, w = e.shape
        e_spa, e_tem = e, e
        if h % self.spa_size != 0 or w % self.spa_size != 0:
            spa_padding = (0, (self.spa_size * int(w / self.spa_size + 1) - w) % self.spa_size,
                           0, (self.spa_size * int(h / self.spa_size + 1) - h) % self.spa_size,
                           )
            e_spa = F.pad(e_spa, spa_padding, mode='constant', value=0)

        if h % self.tem_size != 0 or w % self.tem_size != 0:
            tem_padding = (0, (self.tem_size * int(w / self.tem_size + 1) - w) % self.tem_size,
                           0, (self.tem_size * int(h / self.tem_size + 1) - h) % self.tem_size,
                           )
            e_tem = F.pad(e_tem, tem_padding, mode='constant', value=0)

        s_spa = rearrange(e_spa, 'b l c (n1 h_p) (n2 w_p) -> (b l) (n1 n2) (c h_p w_p)', h_p=self.spa_size,
                          w_p=self.spa_size)
        s_tem = rearrange(e_tem, 'b l c (n1 h_p) (n2 w_p) -> (b n1 n2) l (c h_p w_p)', h_p=self.tem_size,
                          w_p=self.tem_size)

        s_spa = self.lin_spa(s_spa)
        s_tem = self.lin_tem(s_tem)
        return s_spa, s_tem


class embeddings_to_features(nn.Module):
    def __init__(self, channel_number=128, patch_size=np.array([3, 7], np.int),
                 patch_number=np.array([26, 12], np.int)):
        super().__init__()
        self.spa_size = patch_size[0]
        self.tem_size = patch_size[1]
        self.channel_number = channel_number
        self.spa_n_h = patch_number[0]
        self.tem_n_h = patch_number[1]
        self.lin_spa = nn.Linear(1024, 1152)
        self.lin_tem = nn.Linear(1024, 6272)

    def forward(self, s_spa, s_tem, e):
        b, t, c, h, w = e.shape
        s_spa = self.lin_spa(s_spa)
        s_tem = self.lin_tem(s_tem)

        s_spa = rearrange(s_spa, '(b l) (n1 n2) (c h_p w_p) -> b l c (n1 h_p) (n2 w_p)',
                          b=b, n1=self.spa_n_h, c=self.channel_number, h_p=self.spa_size)
        s_tem = rearrange(s_tem, '(b n1 n2) l (c h_p w_p) -> b l c (n1 h_p) (n2 w_p)',
                          b=b, n1=self.tem_n_h, c=self.channel_number, h_p=self.tem_size)

        s_spa, s_tem = s_spa[..., :h, :w], s_tem[..., :h, :w]
        return s_spa + s_tem


class polar_location_encoding(nn.Module):
    def __init__(self, patch_number=264, seq_length=5):
        super().__init__()
        self.d, self.a = nn.Parameter(torch.zeros(patch_number)), nn.Parameter(torch.zeros(patch_number))
        self.c, self.b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(seq_length))
        self.g_l_spa = torch.tensor(np.zeros([patch_number, patch_number]))
        for i in range(patch_number):
            for j in range(patch_number):
                self.g_l_spa[i, j] = (self.d[i] ** 2 + self.d[j] ** 2 - self.d[i] * self.d[j] * self.a[
                    abs(i - j)]) ** 0.5
        self.g_l_tem = torch.tensor(np.zeros([seq_length, seq_length]))
        for i in range(seq_length):
            for j in range(seq_length):
                self.g_l_tem[i, j] = self.c * ((self.b[i] ** 2 + self.b[j] ** 2) ** 0.5)

    def forward(self):
        return self.g_l_spa, self.g_l_tem


class trajectory_aware_attention(nn.Module):
    def __init__(self, dim_seq=1024, num_heads=8, dim_head=128, seq_length=16):
        super().__init__()
        dim_inner = dim_head * num_heads
        self.to_qkv = nn.Linear(dim_seq, dim_inner * 3, bias=False)
        self.to_qk = nn.Linear(dim_seq, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_seq)

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

    def forward(self, s_v, s_m, g_l):
        b, n_l, _, h = *s_v.shape, self.num_heads

        qkv_v = self.to_qkv(s_v).chunk(3, dim=-1)
        qk_m = self.to_qk(s_m).chunk(2, dim=-1)

        q_v, k_v, v = map(lambda t: rearrange(t, 'b nw_l (h d) -> b h nw_l d', h=h), qkv_v)
        q_m, k_m = map(lambda t: rearrange(t, 'b nw_l (h d) -> b h nw_l d', h=h), qk_m)

        p_m = einsum('b h i d, b h j d -> b h i j', q_m, k_m).softmax(dim=-1)
        p_v = einsum('b h i d, b h j d -> b h i j', q_v, k_v)
        dots = p_m[..., :, :] * p_v[..., :, :] * self.scale + g_l.to(torch.float32)

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h nw_l d -> b nw_l (h d)')
        out = self.to_out(out)

        return out


class Residual_Connection(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Layer_Normal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class transformer_block(nn.Module):
    def __init__(self, dim_seq=1024, dim_mlp=1024):
        super().__init__()
        self.dim_seq = dim_seq
        self.attention_block = trajectory_aware_attention()
        self.mlp_block = Residual_Connection(Layer_Normal(dim_seq, MLP_Block(dim=dim_seq, hidden_dim=dim_mlp)))

    def forward(self, s_v, s_m, opt):
        s_v, s_m = nn.LayerNorm(self.dim_seq)(s_v), nn.LayerNorm(self.dim_seq)(s_m)
        s_out = self.attention_block(s_v, s_m, opt) + s_v
        s_out = self.mlp_block(s_out)
        return s_out


class active_clue_aggregator(nn.Module):
    def __init__(self, channel_number=128):
        super().__init__()
        self.ori_gate = nn.Sequential(
            nn.Conv2d(in_channels=channel_number * 2, out_channels=channel_number, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.rtv_gate = nn.Sequential(
            nn.Conv2d(in_channels=channel_number * 2, out_channels=channel_number, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.ori_h, self.rtv_h = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_channels=channel_number * 2, out_channels=channel_number, kernel_size=3, stride=1,
                                  padding=1)

    def forward(self, e_ori, e_rtv):
        e_cat = torch.cat((e_ori, e_rtv), dim=1)
        w_ori, w_rtv = self.ori_h * self.ori_gate(e_cat), self.rtv_h * self.rtv_gate(e_cat)
        e_out = torch.cat((w_ori * e_ori + e_rtv, w_rtv * e_rtv + e_ori), dim=1)
        return self.out_conv(e_out)


class TACNet_Generator(BaseNetwork):
    def __init__(self, input_channel=3, seq_length=10,
                 encoder_channel=np.array([64, 64, 128, 128, 256], dtype=np.int),
                 encoder_group=np.array([1, 2, 4, 8, 1], dtype=np.int),
                 decoder_channel=np.array([128, 64, 64, 3], dtype=np.int),
                 feat2emb_size=np.array([3, 7], np.int),
                 trans_dim_seq=1024, trans_dim_mlp=1024,
                 patch_number=364,
                 init_weights=True):
        super(TACNet_Generator, self).__init__()

        self.ori_encoder1 = feature_encoder(input_channel=input_channel, feature_channel=encoder_channel)
        self.ori_encoder2 = hierarchy_encoder(input_channel=256, hierarchy_group=encoder_group)
        self.ori_decoder = decoder(input_channel=128, feature_channel=decoder_channel)
        self.rtv_encoder1 = feature_encoder(input_channel=input_channel, feature_channel=encoder_channel)
        self.rtv_encoder2 = hierarchy_encoder(input_channel=256, hierarchy_group=encoder_group)
        self.rtv_decoder = decoder(input_channel=128, feature_channel=decoder_channel)
        self.msk_encoder1 = feature_encoder(input_channel=input_channel, feature_channel=encoder_channel)
        self.msk_encoder2 = hierarchy_encoder(input_channel=256, hierarchy_group=encoder_group)
        self.feat2emb = features_to_embeddings(patch_size=feat2emb_size)
        self.emb2feat = embeddings_to_features(patch_size=feat2emb_size)
        self.transformers = nn.Sequential(transformer_block(dim_seq=trans_dim_seq, dim_mlp=trans_dim_mlp),
                                          transformer_block(dim_seq=trans_dim_seq, dim_mlp=trans_dim_mlp),
                                          transformer_block(dim_seq=trans_dim_seq, dim_mlp=trans_dim_mlp),
                                          transformer_block(dim_seq=trans_dim_seq, dim_mlp=trans_dim_mlp), )
        self.clue_agg = active_clue_aggregator(channel_number=128)
        self.polar_encoding = polar_location_encoding(patch_number=patch_number, seq_length=seq_length)

    def forward(self, ori_seq, rtv_seq, msk_seq):

        b, t, c, h, w = ori_seq.size()
        # Two-stage Encoder
        e_ori, e_rtv, e_msk = None, None, None
        for i in range(t):
            tmp_ori = self.ori_encoder2(self.ori_encoder1(ori_seq[:, i]))
            tmp_rtv = self.rtv_encoder2(self.rtv_encoder1(rtv_seq[:, i]))
            tmp_msk = self.msk_encoder2(self.msk_encoder1(msk_seq[:, i]))
            tb, tc, th, tw = tmp_ori.shape
            tmp_ori = tmp_ori.view(tb, -1, tc, th, tw)
            tmp_rtv = tmp_rtv.view(tb, -1, tc, th, tw)
            tmp_msk = tmp_msk.view(tb, -1, tc, th, tw)
            if i != 0:
                e_ori = torch.cat((e_ori, tmp_ori), dim=1)
                e_rtv = torch.cat((e_rtv, tmp_rtv), dim=1)
                e_msk = torch.cat((e_msk, tmp_msk), dim=1)
                continue
            e_ori, e_rtv, e_msk = tmp_ori, tmp_rtv, tmp_msk
        # Trajectory-aware Transformer Module
        s_spa_ori, s_tem_ori = self.feat2emb(e_ori)
        s_spa_rtv, s_tem_rtv = self.feat2emb(e_rtv)
        s_spa_msk, s_tem_msk = self.feat2emb(e_msk)

        g_l_spa, g_l_tem = self.polar_encoding()

        for seq2seq_layer in self.transformers:
            s_spa_ori = seq2seq_layer(s_spa_ori, s_spa_msk, g_l_spa)
            s_tem_ori = seq2seq_layer(s_tem_ori, s_tem_msk, g_l_tem)
            s_spa_rtv = seq2seq_layer(s_spa_rtv, s_spa_msk, g_l_spa)
            s_tem_rtv = seq2seq_layer(s_tem_rtv, s_tem_msk, g_l_tem)

        e_ori_de = self.emb2feat(s_spa_ori, s_tem_ori, e_ori)
        e_rtv_de = self.emb2feat(s_spa_rtv, s_tem_rtv, e_rtv)

        # One-stage Decoder
        ret_ori, ret_rtv = None, None
        for i in range(t):

            tmp_ori, tmp_rtv = e_ori_de[:, i], e_rtv_de[:, i]
            tmp_ori, tmp_rtv = self.clue_agg(tmp_ori, tmp_rtv), tmp_rtv
            tmp_ori = self.ori_decoder(tmp_ori)
            tmp_rtv = self.rtv_decoder(tmp_rtv)

            tb, tc, th, tw = tmp_ori.shape
            tmp_ori = tmp_ori.view(tb, -1, tc, th, tw)
            tmp_rtv = tmp_rtv.view(tb, -1, tc, th, tw)
            if i != 0:
                ret_ori = torch.cat((ret_ori, tmp_ori), dim=1)
                ret_rtv = torch.cat((ret_rtv, tmp_rtv), dim=1)
                continue
            ret_ori, ret_rtv = tmp_ori, tmp_rtv

        return ret_ori, ret_rtv


class SpectralNormStateDictHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


class SpectralNormLoadStateDictPreHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']


class SpectralNorm(object):
    _version = 1

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):

        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def _spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


class TACNet_Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(TACNet_Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def find_l1_l2(img):
    tmp = np.sum(img, axis=0)
    l1, l2 = None, None
    for i in range(tmp.shape[0]):
        if tmp[i] == 0:
            l1 = i
            break
    for i in range(tmp.shape[0] - 1, -1, -1):
        if tmp[i] == 0:
            l2 = i
            break
    return l1, l2


def trajectory_attention_maps(img):
    h, w = img.shape
    l1, l2 = find_l1_l2(img)
    cen = int((l1 + l2) / 2)
    tmp_w = np.zeros([h, w], dtype=np.float)
    for i in range(int(w / 2)):
        if i > int(l2 - l1) / 2:
            tmp_w[:, int(w / 2 + i)] = (w / 2 - i) / (w / 2)
            tmp_w[:, int(w / 2 - 1 - i)] = (w / 2 - i) / (w / 2)
    tmp_h = np.zeros([h, w], dtype=np.float)
    for i in range(int(h / 2)):
        tmp_h[int(h / 2 + i)] = (h / 2 - i) / (h / 2)
        tmp_h[int(h / 2 - 1 - i)] = (h / 2 - i) / (h / 2)
    tmp = ((tmp_w * tmp_h) * 255)
    tmp = tmp.astype(np.uint8)
    ret = np.zeros([h, w], dtype=np.uint8)
    if cen < w / 2:
        rag = cen + int(w / 2)
        ret[:, :rag] = tmp[:, w - rag:]
        ret[:, rag:w] = tmp[:, :w - rag]
    else:
        rag = cen - int(w / 2)
        ret[:, rag:w] = tmp[:, :w - rag]
        ret[:, :rag] = tmp[:, w - rag:]
    return torch.from_numpy(ret)