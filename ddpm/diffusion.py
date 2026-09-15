import math
import copy
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
import random
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as _ckpt_fn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM

def exists(x):
    return x is not None
def noop(*args, **kwargs):
    pass
def is_odd(n):
    return (n % 2) == 1
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def cycle(dl):
    while True:
        for data in dl:
            yield data
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype if x.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SinusoidalPosEmb_small(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype if x.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = math.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
        learned_variance=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.learned_variance = learned_variance
        self.use_checkpoint = use_checkpoint
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)
        print(f"[Unet3D] dim={dim} cond_dim={cond_dim} out_dim={out_dim} attn_heads={attn_heads} attn_dim_head={attn_dim_head} learned_variance={learned_variance}")
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        
        if self.has_cond and not use_bert_text_cond:
            self.time_interval_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
            )
            self.blur_x_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
            )
            self.blur_y_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
            )
            self.blur_z_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
            )
            self.null_cond_emb   = nn.Parameter(torch.randn(1, cond_dim))
            self.null_blur_x_emb = nn.Parameter(torch.randn(1, cond_dim))
            self.null_blur_y_emb = nn.Parameter(torch.randn(1, cond_dim))
            self.null_blur_z_emb = nn.Parameter(torch.randn(1, cond_dim))
            self.age_mlp = nn.Sequential(
                SinusoidalPosEmb_small(dim),
                nn.Linear(dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
            )
            self.null_age_emb = nn.Parameter(torch.randn(1, cond_dim))

        else:
            self.time_interval_mlp = None
            self.blur_mlp = None
            self.null_cond_emb = None
            self.null_blur_emb = None
            self.age_mlp = None
            self.null_age_emb = None

        cond_dim = time_dim + int(cond_dim or 0) * 5  # dt emb + blur emb + age emb
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        out_dim = default(out_dim, channels)
        model_out_dim = out_dim * (2 if learned_variance else 1)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, model_out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ): 
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits
        # null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        # return null_logits + (logits - null_logits) * cond_scale
        if self.learned_variance:
            # Split mean and variance, only scale the mean
            pred_mean, pred_var_logits = logits.chunk(2, dim=1)
            null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
            null_mean, _ = null_logits.chunk(2, dim=1)
            scaled_mean = null_mean + (pred_mean - null_mean) * cond_scale
            return torch.cat([scaled_mean, pred_var_logits], dim=1)
        else:
            null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
            return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.
    ):
        batch, device = x.shape[0], x.device
        use_ckpt = self.use_checkpoint and self.training

        try:
            param_dtype = next(self.parameters()).dtype
        except StopIteration:
            param_dtype = x.dtype

        if x.dtype != param_dtype:
            x = x.to(param_dtype)
        if not time.is_floating_point() or time.dtype != param_dtype:
            time = time.to(param_dtype)
        if cond is not None and cond.is_floating_point() and cond.dtype != param_dtype:
            cond = cond.to(param_dtype)

        if cond is None and self.has_cond:
            cond = torch.zeros((batch, 5), device=device, dtype=param_dtype)  # [dt, blur_x, blur_y, blur_z, age]
           
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if self.has_cond and self.time_interval_mlp is not None:
            cond = cond.to(device)

            dt = cond[:, 0].view(-1)    # ensure [B]
            blur_x = cond[:, 1].view(-1)  # ensure [B]
            blur_y = cond[:, 2].view(-1)  # ensure [B]
            blur_z = cond[:, 3].view(-1)  # ensure [B]
            age = cond[:, 4].view(-1)   # ensure [B]

            cond_emb = self.time_interval_mlp(dt)  # [B, cond_dim]
            mask = prob_mask_like((batch,), null_cond_prob, device=device)

            cond_emb = torch.where(rearrange(mask, 'b -> b 1'),
                self.null_cond_emb, self.time_interval_mlp(dt))

            blur_x_emb = torch.where(rearrange(mask, 'b -> b 1'),
                self.null_blur_x_emb, self.blur_x_mlp(blur_x))

            blur_y_emb = torch.where(rearrange(mask, 'b -> b 1'),
                self.null_blur_y_emb, self.blur_y_mlp(blur_y))

            blur_z_emb = torch.where(rearrange(mask, 'b -> b 1'),
                self.null_blur_z_emb, self.blur_z_mlp(blur_z))

            age_emb = torch.where(rearrange(mask, 'b -> b 1'),
                self.null_age_emb, self.age_mlp(age))

            t = torch.cat((t, cond_emb, blur_x_emb, blur_y_emb, blur_z_emb, age_emb), dim=-1)

        # def _run_down(block1, block2, spatial_attn, temporal_attn): #OUT of MEM HPC
        #     def fn(x_in, t_in):
        #         x_ = block1(x_in, t_in)
        #         x_ = block2(x_, t_in)
        #         x_ = spatial_attn(x_)
        #         x_ = temporal_attn(x_, pos_bias=time_rel_pos_bias,
        #                            focus_present_mask=focus_present_mask)
        #         return x_
        #     return fn
        def _run_down(block1, block2, spatial_attn, temporal_attn):
            def fn(x_in, t_in):
                x_ = _ckpt_fn(block1, x_in, t_in, use_reentrant=False) if use_ckpt else block1(x_in, t_in)
                x_ = _ckpt_fn(block2, x_, t_in, use_reentrant=False) if use_ckpt else block2(x_, t_in)
                x_ = _ckpt_fn(spatial_attn, x_, use_reentrant=False) if use_ckpt else spatial_attn(x_)
                x_ = _ckpt_fn(
                    lambda x__: temporal_attn(x__, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask),
                    x_, use_reentrant=False
                ) if use_ckpt else temporal_attn(x_, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
                return x_
            return fn

        def _run_mid(x_in, t_in):
            x_ = self.mid_block1(x_in, t_in)
            x_ = self.mid_spatial_attn(x_)
            x_ = self.mid_temporal_attn(
                x_, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x_ = self.mid_block2(x_, t_in)
            return x_

        def _run_up(block1, block2, spatial_attn, temporal_attn):
            def fn(x_in, t_in):
                x_ = block1(x_in, t_in)
                x_ = block2(x_, t_in)
                x_ = spatial_attn(x_)
                x_ = temporal_attn(x_, pos_bias=time_rel_pos_bias,
                                   focus_present_mask=focus_present_mask)
                return x_
            return fn

        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            fn = _run_down(block1, block2, spatial_attn, temporal_attn)
            if use_ckpt:
                x = _ckpt_fn(fn, x, t, use_reentrant=False)
            else:
                x = fn(x, t)
            h.append(x)
            x = downsample(x)

        if use_ckpt:
            x = _ckpt_fn(_run_mid, x, t, use_reentrant=False)
        else:
            x = _run_mid(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            fn = _run_up(block1, block2, spatial_attn, temporal_attn)
            if use_ckpt:
                x = _ckpt_fn(fn, x, t, use_reentrant=False)
            else:
                x = fn(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion_Nolatent(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=2,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        device=None,
        use_guide=True,
        learned_variance=False,
        
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.learned_variance = learned_variance
        self.denoise_fn = denoise_fn
        self.device=device
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        print("timesteps : ", timesteps)
        self.loss_type = loss_type
        self.use_guide = use_guide


        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        register_buffer('log_betas', torch.log(betas))
        register_buffer('log_one_minus_betas', torch.log(1. - betas))


        self.text_use_bert_cls = text_use_bert_cls

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        if isinstance(self.denoise_fn, torch.nn.DataParallel):
            model_output = self.denoise_fn.module.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        else:
            model_output = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
            
        # Handle chunking if we are learning variance
        if self.learned_variance:
            noise, pred_noise_std_logits = model_output.chunk(2, dim=1)
            # 1. Calculate the standard deviation and add epsilon for stability
            step_std = torch.sigmoid(pred_noise_std_logits) + 1e-6 
        else:
            noise = model_output
            step_std = None # Fallback if not learning variance
            
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
            
        # 2. Return step_std as the 4th variable!
        return model_mean, posterior_variance, posterior_log_variance, step_std
    
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_ = x.shape
        
        # 3. Catch the 4th variable here (step_std)
        model_mean, model_variance, model_log_variance, step_std = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
            
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred_image = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
        # 4. Return the denoised image AND the step_std
        return pred_image, step_std
    
    def p_sample_loop_pair(
        self, shape_img0, shape_img1, cond=None, cond_scale=1., device=None,
        image=None, seed=42, variance_start_step=None, enable_memory_optim=True, clear_cache_interval=50,
    ):
        if variance_start_step is None:
            variance_start_step = int(self.num_timesteps * 0.05)

        b = shape_img0[0]
        generator = torch.Generator(device=device).manual_seed(seed)
        img = torch.randn(shape_img0, device=device, generator=generator)
        mask = torch.randn(shape_img1, device=device, generator=generator)
        input = torch.cat((img, mask), dim=1)

        # Memory optimization: periodically clear cache
        accumulated_variance = torch.zeros_like(input, device=device)
        
        if enable_memory_optim and device.type == 'cuda':
            torch.cuda.empty_cache()
        real_img = image

        i = self.num_timesteps - 1
        while i >= 0:
            # Periodically clear GPU cache to prevent OOM
            if enable_memory_optim and (self.num_timesteps - 1 - i) % clear_cache_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            if self.use_guide is not None:
                t_tensor = torch.full((b,), i, dtype=torch.long, device=device)

                # The standard step: always no_grad
                with torch.no_grad():
                    real_noisy_image = self.q_sample(x_start=real_img, t=t_tensor)
                    input[:, 0, :, :, :] = real_noisy_image[:, 0, :, :, :].clone()
                    input_next, step_std = self.p_sample(
                        input, t_tensor, cond=cond, cond_scale=cond_scale
                    )

                # Variance propagation
                if self.learned_variance and i <= variance_start_step:
                    beta_t = extract(self.betas, t_tensor, input.shape)
                    alpha_t = 1. - beta_t
                    alpha_cumprod_t = extract(self.alphas_cumprod, t_tensor, input.shape)
                    posterior_var_t = extract(self.posterior_variance, t_tensor, input.shape)

                    a_t = 1. / torch.sqrt(alpha_t)
                    b_t = -beta_t / (torch.sqrt(alpha_t) * torch.sqrt(1. - alpha_cumprod_t))
                    nonzero_mask = (1. - (t_tensor == 0).float()).reshape(
                        -1, *((1,) * (len(input.shape) - 1))
                    )
                    sigma_sq = nonzero_mask * posterior_var_t
                    accumulated_variance = (
                        (a_t ** 2) * accumulated_variance
                        + (b_t ** 2) * (step_std ** 2)
                        + sigma_sq
                    )

                input = input_next

            i -= 1

        return input, accumulated_variance
    

    @torch.inference_mode()
    def p_sample_loop_pair_ddim(
        self,
        shape_img0,
        shape_img1,
        cond=None,
        cond_scale=1.,
        device=None,
        image=None,
        seed=42,
        skip_interval=10,
        variance_start_step=None,
        eta=0.0,
        clip_denoised=True,
        enable_memory_optim=True,
        clear_cache_interval=10,
    ):
        """
        DDIM sampler with variance propagation.

        eta parameter
            eta=0.0 -> deterministic DDIM (no stochastic noise injection)
            eta=1.0 -> equivalent to DDPM (full stochastic injection); only exact with skip_interval=1
            eta in (0, 1) -> hybrid sampler

        Variance recursion stays consistent with the chosen eta:
        when eta > 0 we inject stochastic noise of variance sigma_t^2 and include it
        in the accumulated variance; when eta = 0 we don't.

        The loop runs transitions s -> s_next for s in [T-1, ..., 1] (DDIM schedule),
        then performs one final denoising step at t=0 to match DDPM's i=0 cleanup
        iteration (without this, samples retain visible high-frequency noise from
        the s=1 -> s_next=0 stochastic injection).
        """
        if variance_start_step is None:
            variance_start_step = int(self.num_timesteps * 0.05)
        print(f" using eta {eta} skip {skip_interval} variance_start_step {variance_start_step}")
        b = shape_img0[0]
        generator = torch.Generator(device=device).manual_seed(seed)
        gen_inject  = torch.Generator(device=device).manual_seed(seed + 1) 

        img = torch.randn(shape_img0, device=device, generator=generator)
        mask = torch.randn(shape_img1, device=device, generator=generator)
        input = torch.cat((img, mask), dim=1)
        guide_noise = torch.randn(shape_img0, device=device, generator=generator)

        accumulated_variance = torch.zeros_like(input, device=device)
        real_img = image

        if enable_memory_optim and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Build DDIM schedule [T-1, T-1-zeta, ..., 0]
        timesteps = list(range(self.num_timesteps - 1, -1, -skip_interval))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # ---------- Main DDIM loop: transitions s -> s_next ----------
        for idx in range(len(timesteps) - 1):
            # Periodically clear GPU cache to prevent OOM
            if enable_memory_optim and idx % clear_cache_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            s = timesteps[idx]
            s_next = timesteps[idx + 1]

            t_tensor      = torch.full((b,), s,      dtype=torch.long, device=device)
            t_next_tensor = torch.full((b,), s_next, dtype=torch.long, device=device)

            with torch.no_grad():
                # Guide trick: enforce noisy ground-truth CT0
                real_noisy = self.q_sample(x_start=real_img, t=t_tensor, noise=guide_noise)
                input[:, 0:1] = real_noisy[:, 0:1].clone()

                # Forward pass: noise mean (+ std if learned_variance)
                if isinstance(self.denoise_fn, torch.nn.DataParallel):
                    model_output = self.denoise_fn.module.forward_with_cond_scale(
                        input, t_tensor, cond=cond, cond_scale=cond_scale)
                else:
                    model_output = self.denoise_fn.forward_with_cond_scale(
                        input, t_tensor, cond=cond, cond_scale=cond_scale)

                if self.learned_variance:
                    eps_mean, eps_std_logits = model_output.chunk(2, dim=1)
                    eps_std = torch.sigmoid(eps_std_logits) + 1e-6
                else:
                    eps_mean = model_output
                    eps_std = None

                # Cumulative alphas at s (current) and s_next (less noisy)
                alpha_cum_s      = extract(self.alphas_cumprod, t_tensor,      input.shape)
                alpha_cum_s_next = extract(self.alphas_cumprod, t_next_tensor, input.shape)

                # Predict clean image
                x_recon = self.predict_start_from_noise(input, t=t_tensor, noise=eps_mean)

                # Apply the SAME clipping logic as DDPM's p_mean_variance.
                # Use a separately-named variable to avoid shadowing the timestep `s`.
                if clip_denoised:
                    if self.use_dynamic_thres:
                        s_thresh = torch.quantile(
                            rearrange(x_recon, 'b ... -> b (...)').abs(),
                            self.dynamic_thres_percentile,
                            dim=-1
                        )
                        s_thresh.clamp_(min=1.)
                        s_thresh = s_thresh.view(-1, *((1,) * (x_recon.ndim - 1)))
                        x_recon = x_recon.clamp(-s_thresh, s_thresh) / s_thresh
                    else:
                        x_recon = x_recon.clamp(-1., 1.)

                # Original DDIM path
                if eta > 0:
                    sigma_t_sq = (eta ** 2) * (
                        (1. - alpha_cum_s_next) / (1. - alpha_cum_s).clamp(min=1e-6)
                    ) * (1. - alpha_cum_s / alpha_cum_s_next.clamp(min=1e-6))
                    sigma_t_sq = sigma_t_sq.clamp(min=0.)

                    direction_coef = torch.sqrt((1. - alpha_cum_s_next - sigma_t_sq).clamp(min=0.))

                    nonzero_mask = (1. - (t_tensor == 0).float()).reshape(
                        -1, *((1,) * (len(input.shape) - 1))
                    )
                    noise = torch.randn(input.shape, dtype=input.dtype, device=device, generator=gen_inject)

                    stochastic = nonzero_mask * torch.sqrt(sigma_t_sq) * noise
                else:
                    sigma_t_sq = torch.zeros_like(input)
                    direction_coef = torch.sqrt(1. - alpha_cum_s_next)
                    stochastic = 0.

                input_next = (
                    torch.sqrt(alpha_cum_s_next) * x_recon
                    + direction_coef * eps_mean
                    + stochastic
                )

                # a_s, b_s for variance propagation (Eq. 20)
                a_s = torch.sqrt(alpha_cum_s_next / alpha_cum_s.clamp(min=1e-6))
                b_s = direction_coef - a_s * torch.sqrt(1. - alpha_cum_s)

            # Variance propagation
            if self.learned_variance and s <= variance_start_step:
        
                accumulated_variance = (
                    (a_s ** 2) * accumulated_variance
                    + (b_s ** 2) * (eps_std ** 2)
                    + sigma_t_sq
                )

            input = input_next

        # ---------- Final denoising step at t=0 (matches DDPM's i=0 cleanup) ----------
        with torch.no_grad():
            t_zero = torch.zeros(b, dtype=torch.long, device=device)

            # Guide trick
            real_noisy = self.q_sample(x_start=real_img, t=t_zero, noise=guide_noise)
            input[:, 0:1] = real_noisy[:, 0:1].clone()

            if isinstance(self.denoise_fn, torch.nn.DataParallel):
                model_output = self.denoise_fn.module.forward_with_cond_scale(
                    input, t_zero, cond=cond, cond_scale=cond_scale)
            else:
                model_output = self.denoise_fn.forward_with_cond_scale(
                    input, t_zero, cond=cond, cond_scale=cond_scale)

            if self.learned_variance:
                eps_mean, eps_std_logits = model_output.chunk(2, dim=1)
                eps_std = torch.sigmoid(eps_std_logits) + 1e-6
            else:
                eps_mean = model_output
                eps_std = None

            x_recon = self.predict_start_from_noise(input, t=t_zero, noise=eps_mean)

            if clip_denoised:
                if self.use_dynamic_thres:
                    s_thresh = torch.quantile(
                        rearrange(x_recon, 'b ... -> b (...)').abs(),
                        self.dynamic_thres_percentile,
                        dim=-1
                    )
                    s_thresh.clamp_(min=1.)
                    s_thresh = s_thresh.view(-1, *((1,) * (x_recon.ndim - 1)))
                    x_recon = x_recon.clamp(-s_thresh, s_thresh) / s_thresh
                else:
                    x_recon = x_recon.clamp(-1., 1.)

            input = x_recon
            
        return input, accumulated_variance

    @torch.inference_mode()
    def p_sample_loop_pair_ddim_A(
        self,
        shape_img0,
        shape_img1,
        cond=None,
        cond_scale=1.,
        device=None,
        image=None,
        seed=42,
        skip_interval=10,
        variance_start_step=None,
        eta=0.0,
        clip_denoised=True,
        enable_memory_optim=True,
        clear_cache_interval=10,
    ):
        """
        DDIM sampler with variance propagation.

        eta parameter
            eta=0.0 -> deterministic DDIM (no stochastic noise injection)
            eta=1.0 -> equivalent to DDPM (full stochastic injection); only exact with skip_interval=1
            eta in (0, 1) -> hybrid sampler

        Variance recursion stays consistent with the chosen eta:
        when eta > 0 we inject stochastic noise of variance sigma_t^2 and include it
        in the accumulated variance; when eta = 0 we don't.

        The loop runs transitions s -> s_next for s in [T-1, ..., 1] (DDIM schedule),
        then performs one final denoising step at t=0 to match DDPM's i=0 cleanup
        iteration (without this, samples retain visible high-frequency noise from
        the s=1 -> s_next=0 stochastic injection).

        Baseline guide noise (CHANGED): the noisy baseline injected at each reverse
        step is now drawn fresh, independently, at every step (Option A) rather than
        reusing a single fixed noise realization across the whole trajectory
        (previous behavior). This matches DiffAtlas's reference implementation,
        which resamples fresh noise per step rather than reusing one draw.
        """
        if variance_start_step is None:
            variance_start_step = int(self.num_timesteps * 0.05)
        print(f" using eta {eta} skip {skip_interval} variance_start_step {variance_start_step}")
        b = shape_img0[0]
        generator = torch.Generator(device=device).manual_seed(seed)
        gen_inject = torch.Generator(device=device).manual_seed(seed + 1)
        gen_guide = torch.Generator(device=device).manual_seed(seed + 2)  # CHANGED: dedicated stream for per-step guide noise

        img = torch.randn(shape_img0, device=device, generator=generator)
        mask = torch.randn(shape_img1, device=device, generator=generator)
        input = torch.cat((img, mask), dim=1)
        # CHANGED: removed the single pre-loop `guide_noise` draw.
        # Guide noise is now resampled fresh inside the loop at each step (see below).

        accumulated_variance = torch.zeros_like(input, device=device)
        real_img = image

        if enable_memory_optim and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Build DDIM schedule [T-1, T-1-zeta, ..., 0]
        timesteps = list(range(self.num_timesteps - 1, -1, -skip_interval))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # ---------- Main DDIM loop: transitions s -> s_next ----------
        for idx in range(len(timesteps) - 1):
            # Periodically clear GPU cache to prevent OOM
            if enable_memory_optim and idx % clear_cache_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            s = timesteps[idx]
            s_next = timesteps[idx + 1]

            t_tensor      = torch.full((b,), s,      dtype=torch.long, device=device)
            t_next_tensor = torch.full((b,), s_next, dtype=torch.long, device=device)

            with torch.no_grad():
                # Guide trick: enforce noisy ground-truth CT0
                # CHANGED: fresh noise draw every step instead of a single reused draw
                guide_noise_t = torch.randn(shape_img0, device=device, generator=gen_guide)
                real_noisy = self.q_sample(x_start=real_img, t=t_tensor, noise=guide_noise_t)
                input[:, 0:1] = real_noisy[:, 0:1].clone()

                # Forward pass: noise mean (+ std if learned_variance)
                if isinstance(self.denoise_fn, torch.nn.DataParallel):
                    model_output = self.denoise_fn.module.forward_with_cond_scale(
                        input, t_tensor, cond=cond, cond_scale=cond_scale)
                else:
                    model_output = self.denoise_fn.forward_with_cond_scale(
                        input, t_tensor, cond=cond, cond_scale=cond_scale)

                if self.learned_variance:
                    eps_mean, eps_std_logits = model_output.chunk(2, dim=1)
                    eps_std = torch.sigmoid(eps_std_logits) + 1e-6
                else:
                    eps_mean = model_output
                    eps_std = None

                # Cumulative alphas at s (current) and s_next (less noisy)
                alpha_cum_s      = extract(self.alphas_cumprod, t_tensor,      input.shape)
                alpha_cum_s_next = extract(self.alphas_cumprod, t_next_tensor, input.shape)

                # Predict clean image
                x_recon = self.predict_start_from_noise(input, t=t_tensor, noise=eps_mean)

                # Apply the SAME clipping logic as DDPM's p_mean_variance.
                # Use a separately-named variable to avoid shadowing the timestep `s`.
                if clip_denoised:
                    if self.use_dynamic_thres:
                        s_thresh = torch.quantile(
                            rearrange(x_recon, 'b ... -> b (...)').abs(),
                            self.dynamic_thres_percentile,
                            dim=-1
                        )
                        s_thresh.clamp_(min=1.)
                        s_thresh = s_thresh.view(-1, *((1,) * (x_recon.ndim - 1)))
                        x_recon = x_recon.clamp(-s_thresh, s_thresh) / s_thresh
                    else:
                        x_recon = x_recon.clamp(-1., 1.)

                # Original DDIM path
                if eta > 0:
                    sigma_t_sq = (eta ** 2) * (
                        (1. - alpha_cum_s_next) / (1. - alpha_cum_s).clamp(min=1e-6)
                    ) * (1. - alpha_cum_s / alpha_cum_s_next.clamp(min=1e-6))
                    sigma_t_sq = sigma_t_sq.clamp(min=0.)

                    direction_coef = torch.sqrt((1. - alpha_cum_s_next - sigma_t_sq).clamp(min=0.))

                    nonzero_mask = (1. - (t_tensor == 0).float()).reshape(
                        -1, *((1,) * (len(input.shape) - 1))
                    )
                    noise = torch.randn(input.shape, dtype=input.dtype, device=device, generator=gen_inject)

                    stochastic = nonzero_mask * torch.sqrt(sigma_t_sq) * noise
                else:
                    sigma_t_sq = torch.zeros_like(input)
                    direction_coef = torch.sqrt(1. - alpha_cum_s_next)
                    stochastic = 0.

                input_next = (
                    torch.sqrt(alpha_cum_s_next) * x_recon
                    + direction_coef * eps_mean
                    + stochastic
                )

                # a_s, b_s for variance propagation (Eq. 20)
                a_s = torch.sqrt(alpha_cum_s_next / alpha_cum_s.clamp(min=1e-6))
                b_s = direction_coef - a_s * torch.sqrt(1. - alpha_cum_s)

            # Variance propagation
            if self.learned_variance and s <= variance_start_step:

                accumulated_variance = (
                    (a_s ** 2) * accumulated_variance
                    + (b_s ** 2) * (eps_std ** 2)
                    + sigma_t_sq
                )

            input = input_next

        # ---------- Final denoising step at t=0 (matches DDPM's i=0 cleanup) ----------
        with torch.no_grad():
            t_zero = torch.zeros(b, dtype=torch.long, device=device)

            # Guide trick
            # CHANGED: fresh noise draw here too, consistent with the main loop
            guide_noise_t = torch.randn(shape_img0, device=device, generator=gen_guide)
            real_noisy = self.q_sample(x_start=real_img, t=t_zero, noise=guide_noise_t)
            input[:, 0:1] = real_noisy[:, 0:1].clone()

            if isinstance(self.denoise_fn, torch.nn.DataParallel):
                model_output = self.denoise_fn.module.forward_with_cond_scale(
                    input, t_zero, cond=cond, cond_scale=cond_scale)
            else:
                model_output = self.denoise_fn.forward_with_cond_scale(
                    input, t_zero, cond=cond, cond_scale=cond_scale)

            if self.learned_variance:
                eps_mean, eps_std_logits = model_output.chunk(2, dim=1)
                eps_std = torch.sigmoid(eps_std_logits) + 1e-6
            else:
                eps_mean = model_output
                eps_std = None

            x_recon = self.predict_start_from_noise(input, t=t_zero, noise=eps_mean)

            if clip_denoised:
                if self.use_dynamic_thres:
                    s_thresh = torch.quantile(
                        rearrange(x_recon, 'b ... -> b (...)').abs(),
                        self.dynamic_thres_percentile,
                        dim=-1
                    )
                    s_thresh.clamp_(min=1.)
                    s_thresh = s_thresh.view(-1, *((1,) * (x_recon.ndim - 1)))
                    x_recon = x_recon.clamp(-s_thresh, s_thresh) / s_thresh
                else:
                    x_recon = x_recon.clamp(-1., 1.)

            input = x_recon

        return input, accumulated_variance

    def q_sample(self, x_start, t, noise=None):
  
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
    
    def q_sample_one_step(self, x_prev, t):
        beta_t = extract(self.betas, t, x_prev.shape)
        alpha_t = 1.0 - beta_t
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_beta_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_prev)
        x_t = sqrt_alpha_t * x_prev + sqrt_beta_t * noise
        return x_t
    

    def p_losses(self, img0_start, t, img1_start, cond=None, noise_img0=None, noise_img1=None, **kwargs):
        device = img0_start.device
        img0_start = img0_start.to(device=device, dtype=torch.float32)
        img1_start = img1_start.to(device=device, dtype=torch.float32)
        
        noise_img0 = default(noise_img0, lambda: torch.randn_like(img0_start))
        noise_img1 = default(noise_img1, lambda: torch.randn_like(img1_start))

        img0_noisy = self.q_sample(x_start=img0_start, t=t, noise=noise_img0)
        img1_noisy = self.q_sample(x_start=img1_start, t=t, noise=noise_img1)

        input = torch.cat((img0_noisy, img1_noisy), dim=1)
       
        recon = self.denoise_fn(**dict(x=input, time=t, cond=cond, **kwargs))
        
        if self.learned_variance:
            # 1. Split output into Mean and Std Logits
            pred_noise_mean, pred_noise_std_logits = recon.chunk(2, dim=1)
            
            # 2. Apply sigmoid to bound Std to (0, 1) and add eps for numerical stability
            pred_noise_std = torch.sigmoid(pred_noise_std_logits) + 1e-6
            
            img0_recon_mean = pred_noise_mean[:, 0:1, :, :, :]
            img1_recon_mean = pred_noise_mean[:, 1:2, :, :, :]
            
            # 3. Simple Denoising Loss (L1 or L2)
            if self.loss_type == 'l1':
                loss_0 = F.l1_loss(noise_img0, img0_recon_mean)
                loss_1 = F.l1_loss(noise_img1, img1_recon_mean)
            elif self.loss_type == 'l2':
                loss_0 = F.mse_loss(noise_img0, img0_recon_mean)
                loss_1 = F.mse_loss(noise_img1, img1_recon_mean)
            else:
                raise NotImplementedError()
                
            loss_simple = loss_0 + loss_1
            
            # 4. NLL Loss with Stop-Gradient
            detached_mean = pred_noise_mean.detach()
            
            # Split the target and std into channel 0 (CT0) and channel 1 (Residual)
            pred_std_0 = pred_noise_std[:, 0:1, :, :, :]
            pred_std_1 = pred_noise_std[:, 1:2, :, :, :]
            
            detached_mean_0 = detached_mean[:, 0:1, :, :, :]
            detached_mean_1 = detached_mean[:, 1:2, :, :, :]

            # NLL for Channel 0 (CT_0)
            nll_term1_0 = torch.log(math.sqrt(2 * math.pi) * pred_std_0)
            nll_term2_0 = ((noise_img0 - detached_mean_0) ** 2) / (2 * (pred_std_0 ** 2))
            loss_nll_0 = (nll_term1_0 + nll_term2_0).mean()

            # NLL for Channel 1 (Residual CT1-CT0)
            nll_term1_1 = torch.log(math.sqrt(2 * math.pi) * pred_std_1)
            nll_term2_1 = ((noise_img1 - detached_mean_1) ** 2) / (2 * (pred_std_1 ** 2))
            loss_nll_1 = (nll_term1_1 + nll_term2_1).mean()
            
            # Combine NLL
            loss_nll_total = loss_nll_0 + loss_nll_1
            
            # 5. Combine All Losses
            lambda_nll = 0.01
            total_loss = loss_simple + lambda_nll * loss_nll_total 
            
            loss_dict = {
                'loss_0': loss_0.item(),
                'loss_1': loss_1.item(),
                'loss_simple': loss_simple.item(),
                'loss_nll_0': loss_nll_0.item(),        # Track CT_0 uncertainty loss
                'loss_nll_1': loss_nll_1.item(),        # Track Residual uncertainty loss
                'loss_nll_total': loss_nll_total.item(),
                'consistency_loss': 0
            }
            
        else:
            # Original Logic fallback
            img0_recon = recon[:, 0:1, :, :, :]
            img1_recon = recon[:, 1:2, :, :, :]

            if self.loss_type == 'l1':
                loss_0 = F.l1_loss(noise_img0, img0_recon)
                loss_1 = F.l1_loss(noise_img1, img1_recon)
            elif self.loss_type == 'l2':
                loss_0 = F.mse_loss(noise_img0, img0_recon)
                loss_1 = F.mse_loss(noise_img1, img1_recon)
            else:
                raise NotImplementedError()

            loss_simple = loss_0 + loss_1
            total_loss = loss_simple
            
            loss_dict = {
                'loss_0': loss_0.item(),
                'loss_1': loss_1.item(),
                'consistency_loss': 0,
            }
            if self.learned_variance:
                loss_dict['pred_noise_std_mean'] = pred_noise_std.mean().item()
                loss_dict['pred_noise_std_min'] = pred_noise_std.min().item()
                loss_dict['pred_noise_std_max'] = pred_noise_std.max().item()
    
        return total_loss, loss_dict

    # Simply forward to p_losses
    def forward(self, img0, img1, *args, **kwargs):
        b, device, img_size, = img0.shape[0], img0.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long().to(self.device)
        return self.p_losses(**dict(img0_start=img0, t=t, img1_start=img1, *args, **kwargs))

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        dataset=None,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        max_grad_norm=None,
        num_workers=4,
        diff=False,
        device=None,
        writer=None,
        use_fsdp=False,
        fsdp_mixed_precision=True,
    ):
        super().__init__()

        self.use_fsdp = use_fsdp
        if use_fsdp:
            assert dist.is_initialized(), \
                "Distributed process group must be initialized before Trainer when use_fsdp=True"
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        self.is_main = (self.rank == 0)

        if use_fsdp:
            mp_policy = None
            if fsdp_mixed_precision:
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                    cast_forward_inputs=True,
                )

            wrap_policy = ModuleWrapPolicy({ResnetBlock})
            inner = diffusion_model.denoise_fn
            if isinstance(inner, nn.DataParallel):
                inner = inner.module

            diffusion_model.denoise_fn = FSDP(
                inner,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mp_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                use_orig_params=True,
                sync_module_states=True,
            )
            diffusion_model = diffusion_model.to(torch.cuda.current_device())

        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        if use_fsdp:
            self.ema_model_cpu = self._build_ema_mirror(cfg)
            self._sync_ema_from_fsdp()
            self.ema_model = None
        else:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model_cpu = None
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.writer = writer if self.is_main else None

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.device = device if device is not None else torch.device(f'cuda:{self.local_rank}')
        self.diff = diff
        self.cfg = cfg

        self.ds = dataset
        g = torch.Generator()
        g.manual_seed(1 + self.rank)

        if use_fsdp:
            self.sampler = DistributedSampler(
                self.ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=1,
                drop_last=True,
            )
            dl = DataLoader(
                self.ds,
                batch_size=train_batch_size,
                sampler=self.sampler,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            self.sampler = None
            dl = DataLoader(self.ds, batch_size=train_batch_size,
                            shuffle=True, pin_memory=True, num_workers=num_workers,
                            prefetch_factor=2 if num_workers > 0 else None,
                            persistent_workers=True if num_workers > 0 else False,
                            worker_init_fn=seed_worker, generator=g)

        self.len_dataloader = len(dl)
        if self.is_main:
            print("len_dl ", len(dl))
        self.dl = cycle(dl)

        if self.is_main:
            print(f'found {len(self.ds)} training samples')
        assert len(self.ds) > 0, 'need to have at least 1 sample'

        self.opt = Adam(self.model.parameters(), lr=train_lr)

        self.step = 0

        if use_fsdp and fsdp_mixed_precision:
            self.amp = False
            self.scaler = GradScaler(enabled=False)
        else:
            self.amp = amp
            self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.results_folder = Path(results_folder)
        if self.is_main:
            self.results_folder.mkdir(exist_ok=True, parents=True)
        if use_fsdp:
            dist.barrier()

        if not use_fsdp:
            self.reset_parameters()

    def _build_ema_mirror(self, cfg):
        conditioning_dim = 256 if cfg.model.use_time_interval else 0
        mirror = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=tuple(cfg.model.dim_mults),
            channels=cfg.model.diffusion_num_channels,
            cond_dim=conditioning_dim,
            learned_variance=cfg.model.learned_variance,
        )
        mirror.eval()
        for param in mirror.parameters():
            param.requires_grad_(False)
        return mirror

    @torch.no_grad()
    def _sync_ema_from_fsdp(self):
        with FSDP.state_dict_type(
            self.model.denoise_fn,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            full_sd = self.model.denoise_fn.state_dict()
        self.ema_model_cpu.load_state_dict(full_sd)
        del full_sd

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            if self.use_fsdp:
                self._sync_ema_from_fsdp()
            else:
                self.reset_parameters()
            return
        if self.use_fsdp:
            with FSDP.state_dict_type(
                self.model.denoise_fn,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                full_sd = self.model.denoise_fn.state_dict()
            ema_sd = self.ema_model_cpu.state_dict()
            with torch.no_grad():
                for key in ema_sd.keys():
                    if key in full_sd:
                        ema_sd[key].mul_(self.ema.beta).add_(
                            full_sd[key].to(device=ema_sd[key].device, dtype=ema_sd[key].dtype),
                            alpha=1 - self.ema.beta,
                        )
            del full_sd
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save_progress(self, milestone, long_term=False):
        if self.use_fsdp:
            cfg_full = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model.denoise_fn, StateDictType.FULL_STATE_DICT, cfg_full
            ):
                model_sd = self.model.denoise_fn.state_dict()

            if self.is_main:
                data = {
                    'step': self.step,
                    'model': model_sd,
                    'ema': self.ema_model_cpu.state_dict(),
                    'scaler': self.scaler.state_dict(),
                }
                if long_term:
                    long_term_path = self.results_folder / 'long_term'
                    long_term_path.mkdir(exist_ok=True)
                    torch.save(data, str(long_term_path / f'model-{milestone}.pt'))
                else:
                    torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
                    all_checkpoints = sorted(
                        self.results_folder.glob('model-*.pt'), key=os.path.getmtime)
                    for old_checkpoint in all_checkpoints[:-2]:
                        old_checkpoint.unlink()
                del data
            dist.barrier()
        else:
            data = {
                'step': self.step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'scaler': self.scaler.state_dict(),
                'optimizer': self.opt.state_dict()
            }
            if long_term:
                long_term_path = self.results_folder / 'long_term'
                long_term_path.mkdir(exist_ok=True)
                torch.save(data, str(long_term_path / f'model-{milestone}.pt'))
            else:
                torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
                all_checkpoints = sorted(self.results_folder.glob('model-*.pt'), key=os.path.getmtime)
                for old_checkpoint in all_checkpoints[:-2]:
                    old_checkpoint.unlink()
            del data
    
        # Force garbage collection after checkpoint
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def save_image(self, img, pred_img, milestone, img_idx):
        """
        Save the middle slice of a 3D volume for visualization.
        Args:
            img: Tensor of shape [B, C, D, H, W] or [C, D, H, W]
            pred_img: Tensor of shape [B, C, D, H, W] or [C, D, H, W]
            milestone: Training milestone number
            img_idx: Image identifier (e.g., 'img0', 'img1')
        """
        # Handle batch dimension
        print(f"Saving image for milestone {milestone}, img_idx {img_idx} with shape {img.shape}")
        if img.dim() == 5:  # [B, C, D, H, W]
            img = img[0]  # Take first sample from batch
            pred_img = pred_img[0]  # Take first sample from batch
        
        # Now img is [C, D, H, W]
        img = img.cpu().numpy()
        pred_img = pred_img.cpu().numpy()
        
        # Get middle slice along depth dimension
        mid_slice = img.shape[1] // 2  # D dimension is at index 1
        
        # Extract middle slice: [C, H, W]
        slice_img = img[:, mid_slice, :, :]
        slice_pred_img = pred_img[:, mid_slice, :, :]
        
        # If multi-channel, take first channel
        if slice_img.shape[0] > 1:
            slice_img = slice_img[0]  # [H, W]
            slice_pred_img = slice_pred_img[0]  # [H, W]
        else:
            slice_img = slice_img.squeeze(0)  # [H, W]
            slice_pred_img = slice_pred_img.squeeze(0)  # [H, W]
        
        # Normalize to [0, 1] for visualization
        slice_img = (slice_img + 1) / 2
        slice_pred_img = (slice_pred_img + 1) / 2

        #clip to [0, 1] just in case
        slice_img = np.clip(slice_img, 0, 1)
        slice_pred_img = np.clip(slice_pred_img, 0, 1)
        
        # Create figure
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(slice_img, cmap='gray')
        plt.title(f'Milestone {milestone} - {img_idx} - Slice {mid_slice} - Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(slice_pred_img, cmap='gray')
        plt.title(f'Milestone {milestone} - {img_idx} - Slice {mid_slice} - Predicted')
        plt.axis('off')
        
        # Save figure
        
        save_path = os.path.join(self.results_folder, 'visualization')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'sample-{milestone}-{img_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved sample image to {save_path}")

    def save_error_and_variance(self, img, pred_img, variance, milestone, img_idx):
        """
        Save a side-by-side heatmap of the absolute error vs the predicted uncertainty.
        Args:
            img: Ground truth tensor [B, C, D, H, W] or [C, D, H, W]
            pred_img: Predicted tensor [B, C, D, H, W] or [C, D, H, W]
            variance: Accumulated variance tensor [B, C, D, H, W] or [C, D, H, W]
            milestone: Training milestone number
            img_idx: Image identifier (e.g., 'residual')
        """
        # Handle batch dimension (take the first sample)
        if img.dim() == 5:
            img = img[0]
            pred_img = pred_img[0]
            variance = variance[0]
            
        # Move to CPU and convert to numpy
        img = img.cpu().numpy()
        pred_img = pred_img.cpu().numpy()
        variance = variance.cpu().numpy()
        
        # Get middle slice along depth dimension
        mid_slice = img.shape[1] // 2 
        
        # We specifically want to look at the residual channel (index 1) if it exists
        c_idx = 1 if img.shape[0] > 1 else 0
        
        slice_img = img[c_idx, mid_slice, :, :]
        slice_pred = pred_img[c_idx, mid_slice, :, :]
        slice_var = variance[c_idx, mid_slice, :, :]
        
        # 1. Calculate Absolute Error
        error_map = np.abs(slice_img - slice_pred)
        
        # 2. Convert Variance to Standard Deviation for visualization
        # (Variance scales quadratically, which makes heatmaps hard to read. Std is linear.)
        std_map = np.sqrt(np.clip(slice_var, a_min=0, a_max=None)) 
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Absolute Error
        # Using 'viridis' colormap: dark is low error, bright yellow/white is high error
        im0 = axes[0].imshow(error_map, cmap='viridis')
        axes[0].set_title(f'Milestone {milestone} - {img_idx} - Absolute Error')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2: Predicted Uncertainty (Standard Deviation)
        # Using 'viridis' colormap to visually distinguish it from the error map
        im1 = axes[1].imshow(std_map, cmap='viridis')
        axes[1].set_title(f'Milestone {milestone} - {img_idx} - Predicted Uncertainty (Std)')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Save figure
        save_path = os.path.join(self.results_folder, 'visualization')
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'sample-{milestone}-{img_idx}_uncertainty.png')
        plt.savefig(save_file, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved uncertainty visualization to {save_file}")


    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)
        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            cand1 = Path(self.results_folder) / f'model-{milestone}.pt'
            cand2 = Path(self.results_folder) / 'long_term' / f'model-{milestone}.pt'
            ckpt_path = cand1 if cand1.exists() else cand2
            if not ckpt_path.exists():
                raise FileNotFoundError(f'No checkpoint at {cand1} or {cand2}')
            if getattr(self, 'is_main', True):
                print(f'[load] reading checkpoint from {ckpt_path}')
            data = torch.load(str(ckpt_path), map_location='cpu')

        self.step = data['step']

        if getattr(self, 'use_fsdp', False):
            with FSDP.state_dict_type(
                self.model.denoise_fn,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                missing, unexpected = self.model.denoise_fn.load_state_dict(
                    data['model'], strict=False)
            if self.is_main:
                if missing:
                    print(f'[load] missing keys: {len(missing)} (first 5: {missing[:5]})')
                if unexpected:
                    print(f'[load] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})')

            if 'ema' in data and self.ema_model_cpu is not None:
                self.ema_model_cpu.load_state_dict(data['ema'], strict=False)
            if 'scaler' in data and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(data['scaler'])
                except Exception as exc:
                    if self.is_main:
                        print(f'[load] skipping scaler state: {exc}')
            if self.is_main:
                print(f'[load] resumed from step {self.step} (Adam moments re-warm under FSDP)')
            dist.barrier()
        else:
            self.model.load_state_dict(data['model'], **kwargs)
            if 'ema' in data and getattr(self, 'ema_model', None) is not None:
                self.ema_model.load_state_dict(data['ema'], **kwargs)
            if 'scaler' in data:
                self.scaler.load_state_dict(data['scaler'])
            if 'optimizer' in data:
                self.opt.load_state_dict(data['optimizer'])

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        epoch = 0
        steps_per_epoch = max(1, self.len_dataloader // max(1, self.gradient_accumulate_every))
        eff_bs = self.batch_size * self.gradient_accumulate_every * self.world_size
        fsdp_mp_active = self.use_fsdp and not self.scaler.is_enabled()

        if self.is_main:
            print(f"[rank {self.rank}] effective batch size = {eff_bs} "
                  f"(local_bs={self.batch_size} x accum={self.gradient_accumulate_every} "
                  f"x world={self.world_size})")

        while self.step < self.train_num_steps:
            if self.use_fsdp and self.sampler is not None and self.step % steps_per_epoch == 0:
                self.sampler.set_epoch(epoch)
                epoch += 1

            total_loss = 0.0
            last_image_0 = None
            last_image_1 = None
            last_cond = None

            for i in range(self.gradient_accumulate_every):

                data_batch = next(self.dl)

                image_0 = data_batch['img0'].to(self.device, non_blocking=True)
                image_1 = data_batch['img1'].to(self.device, non_blocking=True)
                dt = data_batch['dt'].to(self.device, non_blocking=True)
                blur0 = data_batch['blur0'].to(self.device, non_blocking=True)
                blur1 = data_batch['blur1'].to(self.device, non_blocking=True)
                diff_blur = blur1 - blur0
                age = data_batch['age0'].to(self.device, non_blocking=True)
                cond = torch.cat([dt, diff_blur, age], dim=1)
                if i == self.gradient_accumulate_every - 1:
                    last_image_0 = image_0.detach().clone()
                    last_image_1 = image_1.detach().clone()
                    last_cond = cond.detach().clone()

                if fsdp_mp_active:
                    loss, loss_dict = self.model(
                        img0=image_0,
                        img1=image_1,
                        cond=cond,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                    )
                    scaled_loss = loss / self.gradient_accumulate_every
                    scaled_loss.backward()
                else:
                    with autocast(enabled=self.amp):
                        loss, loss_dict = self.model(
                            img0=image_0,
                            img1=image_1,
                            cond=cond,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask,
                        )
                        scaled_loss = loss / self.gradient_accumulate_every
                        self.scaler.scale(scaled_loss).backward()

                total_loss += loss.item()
                del loss, scaled_loss
                del data_batch, image_0, image_1
                if cond is not None:
                    del cond

            loss_value = total_loss / self.gradient_accumulate_every
            if self.is_main:
                print(f'{self.step}: {loss_value}')

            if not math.isfinite(loss_value):
                if self.is_main:
                    print(f'[skip] non-finite loss at step {self.step}: {loss_value}')
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                continue

            log = {'loss': loss_value}
            log.update(loss_dict)

            if fsdp_mp_active:
                if exists(self.max_grad_norm):
                    self.model.denoise_fn.clip_grad_norm_(self.max_grad_norm)
                self.opt.step()
            else:
                if exists(self.max_grad_norm):
                    self.scaler.unscale_(self.opt)
                    if self.use_fsdp:
                        self.model.denoise_fn.clip_grad_norm_(self.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

            self.opt.zero_grad(set_to_none=True)

            if self.step % 10 == 0:
                torch.cuda.empty_cache()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                save_long_term = self.step % 5000 == 0 

                if save_long_term and self.is_main:
                    
                    sampling_model = self._get_sampling_model()
                    sampling_model.eval()
                    try:
                        with torch.no_grad():
                            sample, accumulated_variance = sampling_model.p_sample_loop_pair_ddim(
                                shape_img0=last_image_0.shape,
                                shape_img1=last_image_1.shape,
                                cond=last_cond,
                                cond_scale=1.,
                                device=self.device,
                                image=last_image_0,
                            )
                    finally:
                        self._release_sampling_model()

                    sample_img0 = sample[:, 0:1, :, :, :].cpu()
                    sample_img1 = sample[:, 1:2, :, :, :].cpu()
                    accumulated_variance0 = accumulated_variance[:, 0:1, :, :, :].cpu()
                    accumulated_variance1 = accumulated_variance[:, 1:2, :, :, :].cpu()

                    self.save_image(last_image_0, sample_img0,
                                    milestone=milestone, img_idx='0')

                    if self.diff:
                        sample_img1 = last_image_0.cpu() + sample_img1 * 2
                        last_image_1 = last_image_0.cpu() + last_image_1.cpu() * 2

                    self.save_image(last_image_1, sample_img1,
                                    milestone=milestone, img_idx='1')

                    if self.model.learned_variance:
                        self.save_error_and_variance(last_image_0, sample_img0,
                                                    accumulated_variance0,
                                                    milestone, img_idx='0')
                        self.save_error_and_variance(last_image_1, sample_img1,
                                                    accumulated_variance1,
                                                    milestone, img_idx='1')

                    del sample, sample_img0, sample_img1
                    del accumulated_variance, accumulated_variance0, accumulated_variance1

                self.save_progress(milestone, long_term=save_long_term)
                if self.is_main:
                    print(f"{self.step}: {loss_value}")

                last_image_0 = last_image_1 = last_cond = None
                torch.cuda.empty_cache()

            if self.step % 10 == 0 and self.is_main:
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/lr": self.opt.param_groups[0]['lr'],
                    }, step=self.step)
                    for key, value in log.items():
                        if key != 'loss':
                            wandb.log({f"train/{key}": value}, step=self.step)
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", loss_value, self.step)
                    self.writer.add_scalar("train/lr", self.opt.param_groups[0]['lr'], self.step)
                    for key, value in log.items():
                        if key != 'loss':
                            self.writer.add_scalar(f"train/{key}", value, self.step)

            log_fn(log)
            self.step += 1

        if self.is_main:
            print('training completed')

    def _get_sampling_model(self):
        if not self.use_fsdp:
            return self.ema_model

        if not hasattr(self, '_sampling_module') or self._sampling_module is None:
            self._sampling_denoiser = self.ema_model_cpu.to(self.device)
            self._orig_denoise_fn = self.model.denoise_fn
            self.model.denoise_fn = self._sampling_denoiser
            self._sampling_module = self.model
        return self._sampling_module

    def _release_sampling_model(self):
        if not self.use_fsdp:
            return
        if hasattr(self, '_sampling_module') and self._sampling_module is not None:
            self.model.denoise_fn = self._orig_denoise_fn
            self._sampling_denoiser.cpu()
            del self._sampling_denoiser
            self._orig_denoise_fn = None
            self._sampling_module = None
            torch.cuda.empty_cache()