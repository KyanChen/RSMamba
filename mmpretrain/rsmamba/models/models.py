import math
from typing import Sequence

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.models import build_2d_sincos_position_embedding

from transformers.models.mamba.modeling_mamba import MambaMixer

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import build_norm_layer, resize_pos_embed, to_2tuple
from mmpretrain.models.backbones.base_backbone import BaseBackbone


@MODELS.register_module()
class RSMamba(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768 // 4,
                'num_layers': 8*2,
                'feedforward_channels': 768 // 2,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768//4,
                'num_layers': 12*2,
                'feedforward_channels': 768//2
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024//4,
                'num_layers': 36,
                'feedforward_channels': 1024//2
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                'embed_dims': 1280//4,
                'num_layers': 48,
                'feedforward_channels': 1280//2
            }),
    }
    OUT_TYPES = {'featmap', 'avg_featmap', 'cls_token', 'raw'}

    def __init__(self,
                 arch='base',
                 pe_type='learnable',
                 # 'forward', 'forward_reverse_mean', 'forward_reverse_gate', 'forward_reverse_shuffle_gate'
                 path_type='forward_reverse_shuffle_gate',
                 cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='avg_featmap',
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(RSMamba, self).__init__(init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.cls_position = cls_position
        self.path_type = path_type

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.num_extra_tokens = 0
        # Set cls token
        if cls_position != 'none':
            if cls_position == 'head_tail':
                self.cls_token = nn.Parameter(torch.zeros(1, 2, self.embed_dims))
                self.num_extra_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
                self.num_extra_tokens = 1
        else:
            self.cls_token = None

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pe_type = pe_type
        if pe_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        elif pe_type == 'sine':
            self.pos_embed = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                temperature=10000,
                cls_token=False)
            # TODO: add cls token
        else:
            self.pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.layers = ModuleList()
        self.gate_layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                state_size=16,
                intermediate_size=self.arch_settings.get('feedforward_channels', self.embed_dims * 2),
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                use_bias=False,
            )
            _layer_cfg.update(layer_cfgs[i])
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.layers.append(MambaMixer(_layer_cfg, i))
            if 'gate' in self.path_type:
                gate_out_dim = 2
                if 'shuffle' in self.path_type:
                    gate_out_dim = 3
                self.gate_layers.append(
                    nn.Sequential(
                        nn.Linear(gate_out_dim*self.embed_dims, gate_out_dim, bias=False),
                        nn.Softmax(dim=-1)
                    )
                )

        self.frozen_stages = frozen_stages
        self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)
        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(RSMamba, self).init_weights()

        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if 'gate' in self.path_type:
                m = self.gate_layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.cls_position == 'head':
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_position == 'tail':
                x = torch.cat((x, cls_token), dim=1)
            elif self.cls_position == 'head+tail':
                x = torch.cat((cls_token[:, :1], x, cls_token[:, 1:]), dim=1)
            elif self.cls_position == 'middle':
                x = torch.cat((x[:, :x.size(1) // 2], cls_token, x[:, x.size(1) // 2:]), dim=1)
            else:
                raise ValueError(f'Invalid cls_position {self.cls_position}')

        if self.pos_embed is not None:
            x = x + self.pos_embed.to(device=x.device)

        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            residual = x
            if 'forward' == self.path_type:
                x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
                x = layer(x)

            if 'forward_reverse_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                x = (forward_x + torch.flip(reverse_x, [1])) / 2

            if 'forward_reverse_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x
            if 'forward_reverse_shuffle_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                mean_shuffle_x = torch.mean(shuffle_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x

            if 'forward_reverse_shuffle_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                x = (forward_x + reverse_x + shuffle_x) / 3
            x = residual + x
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            if self.cls_position == 'head':
                return x[:, 0]
            elif self.cls_position == 'tail':
                return x[:, -1]
            elif self.cls_position == 'head_tail':
                x = torch.mean(x[:, [0, -1]], dim=1)
                return x
            elif self.cls_position == 'middle':
                return x[:, x.size(1) // 2]
        patch_token = x
        if self.cls_token is not None:
            if self.cls_position == 'head':
                patch_token = x[:, 1:]
            elif self.cls_position == 'tail':
                patch_token = x[:, :-1]
            elif self.cls_position == 'head_tail':
                patch_token = x[:, 1:-1]
            elif self.cls_position == 'middle':
                patch_token = torch.cat((x[:, :x.size(1) // 2], x[:, x.size(1) // 2 + 1:]), dim=1)
        if self.out_type == 'featmap':
            B = x.size(0)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))