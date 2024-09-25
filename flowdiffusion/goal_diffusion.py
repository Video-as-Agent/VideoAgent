import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import wandb
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import imageio
from torch.optim import Adam
import torch.distributions as dist
from torch.distributed import barrier
from torch.utils.data.distributed import DistributedSampler


import json
from torchvision import transforms as T, utils
import datetime

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from accelerate.logging import get_logger

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

from feedback_binary_rf import chat_with_openai_rf
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

__version__ = "0.0"

from torch.utils.data.dataloader import default_collate

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import tensorboard as tb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def sample_timesteps(b, num_timesteps, device):
    alpha = 5.0  # Parameter for the beta distribution
    beta = 2.0   # Parameter for the beta distribution

    # Create a beta distribution
    beta_dist = dist.Beta(alpha, beta)

    # Sample from the beta distribution
    sampled = beta_dist.sample((b,)).to(device)

    # Scale the samples to the range [0, num_timesteps - 1]
    timesteps = (sampled * (num_timesteps - 1)).long()

    return timesteps
# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, vid_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False, guidance_weight=0):
        # task_embed = self.text_encoder(goal).last_hidden_state
        ##x_2 shape = 8,3,7,48,64
        ##x_cond shape = 8, 3, 48, 64
        f = x.shape[1] // 3
        # print("model pred. x shape", x.shape)
        # print("model_predictions vid_cond shape", vid_cond.shape)
        x_1 = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        if vid_cond.dim() == 4:
            vid_cond = repeat(vid_cond, "b c h w -> b c f h w", f = f)
        # else:
        #     vid_cond = rearrange(vid_cond, 'b f c h w -> b c f h w')
        # print("concatenated x_1", )
        x_1 = torch.cat([x_1, vid_cond], dim=1) ##[8,6,7,128,128]
        # print("model predictions x shape", x_1.shape)
        # print("model predictions", x_1.shape)
        model_out = self.model(x_1, t, task_embed)
        model_output = rearrange(model_out, 'b c f h w -> b (f c) h w')
        
        # model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)
        if guidance_weight > 0.0:
            uncond_model_output = self.model(x_1, t, task_embed*0.0)
            uncond_model_output = rearrange(uncond_model_output, 'b c f h w -> b (f c) h w')

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            if guidance_weight == 0:
                pred_noise = model_output
            else:
                pred_noise = (1 + guidance_weight)*model_output - guidance_weight*uncond_model_output

            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)

            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_x_start = uncond_model_output
                uncond_x_start = maybe_clip(uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            
            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_v = uncond_model_output
                uncond_x_start = self.predict_start_from_v(x, t, uncond_v)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, vid_cond, task_embed,  clip_denoised=False, guidance_weight=0):
        # print("p_mean variance vid_cond shape", vid_cond.shape)
        preds = self.model_predictions(x, t, vid_cond, task_embed, guidance_weight=guidance_weight)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, vid_cond, task_embed, guidance_weight=0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        # print("p_sample vid_cond shape", vid_cond.shape)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, vid_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, vid_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            # print("p_sample loop vid_cond shape", vid_cond.shape)
            img, x_start = self.p_sample(img, t, vid_cond, task_embed, guidance_weight=guidance_weight)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, vid_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            # print("ddim vid_cond shape", vid_cond.shape)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, vid_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, vid_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        image_size, channels = self.image_size, self.channels
        # print("sample vid_cond shape", vid_cond.shape)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]), vid_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    # @property
    # def loss_fn(self):
    #     if self.loss_type == 'l1':
    #         return F.l1_loss
    #     elif self.loss_type == 'l2':
    #         # Define a custom function for the combined loss
    #         def combined_loss(output, target, reduction):
    #             mse_loss = F.mse_loss(output, target, reduction)
    #             l1_loss = F.l1_loss(output, target,)
    #             return mse_loss + 0.01 * l1_loss
            
    #         return combined_loss
    #     else:
    #         raise ValueError(f'Invalid loss type {self.loss_type}')
    
    # b, c, h, w = x_start.shape
    #     noise = default(noise, lambda: torch.randn_like(x_start))

    #     # noise sample

    #     x = self.q_sample(x_start=x_start, t=t, noise=noise)

    #     # predict and take gradient step

    #     model_out = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)

    #     if self.objective == 'pred_noise':
    #         target = noise
    #     elif self.objective == 'pred_x0':
    #         target = x_start
    #     elif self.objective == 'pred_v':
    #         v = self.predict_v(x_start, t, noise)
    #         target = v
    #     else:
    #         raise ValueError(f'unknown objective {self.objective}')

    #     loss = self.loss_fn(model_out, target, reduction = 'none')
    #     loss = reduce(loss, 'b ... -> b (...)', 'mean')

    #     loss = loss * extract(self.loss_weight, t, loss.shape)
    #     return loss.mean()


    def p_losses(self, x_start, t, x_cond, vid_cond, task_embed, noise=None): ##frame conditioning
        
        b, c, h, w = x_start.shape
        # print("x start shape", x_start.shape)
        # print("vid cond shape", vid_cond.shape)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        f = x.shape[1] // 3
        # x_new = x[:,3:,:,:]
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        # print("p_losses_x_cond shape", x_cond.shape)
        # x_cond = repeat(x_cond, 'b c h w -> b c f h w', f = 6)
        # print("x:",x.shape)
        # print("v",vid_cond.shape)
        if vid_cond.shape[1] == f:
            vid_cond = rearrange(vid_cond, 'b f c h w -> b c f h w') ##comment for ithor training
        x = torch.cat([vid_cond, x], dim=1) ##[32,6,7,48,64]
        # x = torch.cat([x, x_cond], dim=1)
        model_out = self.model(x, t, task_embed)
        model_out = rearrange(model_out, 'b c f h w -> b (f c) h w')
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        # Calculate L2 regularization
        l2_reg = torch.tensor(0., device=model_out.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)

        # Combine main loss and regularization
        lambda_reg = 1e-2  # Regularization strength, adjust as needed
        total_loss = loss.mean() + lambda_reg * l2_reg

        return total_loss

    def forward(self, img, img_cond, vid_cond, task_embed): ##frame conditioning
    # def forward(self, img, img_cond, task_embed):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # t = sample_timesteps(b=b, num_timesteps=self.num_timesteps, device=device)

        img = self.normalize(img)
        # return self.p_losses(img, t, img_cond, task_embed)
        return self.p_losses(img, t, img_cond, vid_cond, task_embed) ##frame conditioning

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        wandb_project="Flow Diffusion",
        wandb_entity="trickyjustice"
    ):
        super().__init__()
        
        # wandb.init(project=wandb_project, entity=wandb_entity)

        self.cond_drop_chance = cond_drop_chance
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # accelerator
        init_handler = InitProcessGroupKwargs()
        init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
        
        from accelerate import DistributedDataParallelKwargs
        # init_train = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_train = 'DDP'
        fsdp_plugin = None
        
        self.accelerator = Accelerator(
            # split_batches = split_batches,
            fsdp_plugin=fsdp_plugin,
            even_batches=False,
            kwargs_handlers=[init_handler],
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        self.channels = channels
        self.update_epoch = 10

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle = True, num_workers=0)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)# dl = dataloader
        # self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, num_workers = 0)


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.accelerator.device)
        # self.accelerator.broadcast(self.ema.state_dict(), from_process=0)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)
            
        # #image and video paths for feedback
        # self.image_paths = {}
        # self.video_paths = {}
        # self.video_dir = "/home/a2soni/AVDC/binary/videos"
        # self.image_dir = "/home/a2soni/AVDC/binary/images"

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        model_path = self.results_folder / f'iteration_1-{milestone}_binary.pt'  # Define the path
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(model_path))  # Use the defined path
        # wandb.save(str(model_path))  # Correctly reference the path here

    def load(self, milestone):
        accelerator = self.accelerator
        device = self.accelerator.device

        data = torch.load(str(self.results_folder / f'iteration_1-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    #     return fid_value
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    def sample(self, vid_cond, batch_text, batch_size=1, guidance_weight=0):
        if not hasattr(self, 'ema') or self.ema is None:
            raise AttributeError("EMA model has not been initialized.")
        print(vid_cond.shape)
        device = self.accelerator.device
        task_embeds = self.encode_batch_text(batch_text)
        return self.ema.ema_model.sample(vid_cond.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        if accelerator.is_main_process:
            wandb.init(project=self.wandb_project, entity = self.wandb_entity)
        else:
            wandb.init(project=self.wandb_project, entity = self.wandb_entity, mode="disabled")
            
            # Calculate steps_per_epoch right after you initialize your DataLoader and other parameters
        total_sequences = len(self.ds)  # assuming self.ds is your dataset
        batch_size = self.batch_size  # assuming self.dl is your DataLoader
        steps_per_epoch = (total_sequences // batch_size) + 1  # integer division ceiling
        save_steps = steps_per_epoch*self.update_epoch
        
        update_buffer = {}  # Buffer to store changes to x_2
        self.image_paths = [None] * total_sequences  # Initialize image paths list
        self.video_paths = [None] * total_sequences  # Initialize video paths list
        self.task_descriptions = [None] * total_sequences  # Initialize task descriptions list
        self.update_task_buffer = {}
        self.data_point_counter = 0
        self.initial_step = self.step
        # print(self.initial_step)


        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    idx, x, x_cond, goal, x_2 = next(self.dl) ##frame conditioning
                    # print("x shape", x.shape)
                    # if x.shape[2] != 7:
                        # continue
                    # print("x_cond shape", x_cond.shape)
                    if x.dim() == 5:
                        x = rearrange(x, 'b c f h w -> b (f c) h w')
                    # print("vid cond shape", x_2.shape)
                    x, x_cond, x_2 = x.to(device), x_cond.to(device), x_2.to(device) ##frame conditioning
                    # print("mounted to device")
                    goal_embed = self.encode_batch_text(goal)
                    goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()


                    with self.accelerator.autocast():
                        # print("HERE")
                        loss = self.model(x, x_cond, x_2, goal_embed) ##frame condidioning
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        
                        # with torch.no_grad():
                        # if accelerator.is_main_process:
                        #     generated_video = self.ema.ema_model.sample(batch_size=batch_size, vid_cond=x_2, task_embed=goal_embed)
                        #     for i in range(len(x)):
                        #             # Store generated video in buffer with (idx, start_idx) as key
                        #         update_buffer[(idx[i], start_idx[i])] = generated_video[i]

                        #     # Save x_cond and new_x_2 to cache directory
                        #     img_path = f'/home/achint/AVDC_2/cache/img_{idx[i]}_{start_idx[i]}.png'
                        #     vid_path = f'/home/achint/AVDC_2/cache/vid_{idx[i]}_{start_idx[i]}.gif'
                        #     # torch.save(x_cond[i], img_path)
                        #     # torch.save(generated_video, vid_path)
                        #     # video_tensor = generated_video.cpu().numpy().astype(np.uint8)  # Convert to numpy array and uint8
                        #     output_1 = rearrange(generated_video, '(f c) h w -> f c h w', c = 3)
                        #     # print(output_1.shape)
                        #     output_1 = torch.cat([img_cond.unsqueeze(0), output_1], dim=0).detach()
                        #     # print("output_1 shape", output_1.shape)
                        #     output_1 = (output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
                        #     # print("video shape", output_1.shape)
                        #     imageio.mimsave(vid_path, output_1, duration=200, loop=1000)
                        #     # imageio.mimsave(vid_path, video_tensor, fps=25)
                            
                        #     # image_path = os.path.join(self.image_dir, f'image_{key}.png')
                        #     imageio.imwrite(img_path, (x_cond[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 
                            
                        #     self.image_paths[idx[i]] = img_path
                        #     self.video_paths[idx[i]] = vid_path
                        #     self.task_descriptions[idx[i]] = goal[i]
                        #     self.update_task_buffer[(idx[i], start_idx[i])] = goal[i]
                
                        # feedback_responses = chat_with_openai_rf(self.image_paths, self.video_paths, self.task_descriptions)
                        # print(feedback_responses)
                        
                        # # Update task descriptions with feedback
                        # for i, feedback in enumerate(feedback_responses):
                        #     if feedback is not None:
                        #         self.update_task_buffer[(idx[i], start_idx[i])] += ', feedback is ' + feedback
                    
                        # Process the update buffer to apply changes to dataset
                        # for (sequence_idx, frame_start_idx), new_x_2 in update_buffer.items():
                        #     sequence_idx = sequence_idx.item() if torch.is_tensor(sequence_idx) else sequence_idx
                        #     frame_start_idx = frame_start_idx.item() if torch.is_tensor(frame_start_idx) else frame_start_idx

                        #     self.ds.update_x_2(sequence_idx, frame_start_idx, new_x_2)
                            
                        # # Clear buffers and paths
                        # update_buffer.clear()
                        # update_task_buffer.clear()
                        # self.clear_paths(self.image_paths)
                        # self.clear_paths(self.video_paths)

                        # # Reset lists
                        # self.image_paths = [None] * total_sequences
                        # self.video_paths = [None] * total_sequences
                        # self.task_descriptions = [None] * total_sequences
                
                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                if accelerator.is_main_process:
                    wandb.log({"loss": total_loss, "loss_scale": accelerator.scaler.get_scale(), "step": self.step})

                scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                accelerator.wait_for_everyone()
                
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.valid_batch_size)
                            ### get val_imgs from self.valid_dl
                            # x_2s = []
                            # xs = []
                            # x_conds = []
                            # task_embeds = []
                            # for i, (idx, x,label, x_2) in enumerate(self.valid_dl):
                            #     xs.append(x)
                            #     # x_conds.append(x_cond.to(device))
                            #     x_2s.append(x_2.to(device))
                            #     task_embeds.append(self.encode_batch_text(label))
                            
                            # with self.accelerator.autocast():
                            #     all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, vid_cond=c, task_embed=e), batches, x_2s, task_embeds))
                        
                        print_gpu_utilization()
                        
                        # gt_xs = torch.cat(xs, dim = 0).detach().cpu() # [batch_size, 3*n, 120, 160]
                        # # make it [batchsize*n, 3, 120, 160]
                        # n_rows = gt_xs.shape[1]
                        # # gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
                        # ### save images
                        # # x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                        # # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
                        # all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                        # all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

                        # gt_first = gt_xs[:, :1]
                        # gt_last = gt_xs[:, -1:]



                        # if self.step == self.save_and_sample_every:
                        #     os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                        #     gt_img = torch.cat([gt_first, gt_last, gt_xs], dim=1)
                        #     gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
                        #     utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows+2)

                        # os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        # pred_img = torch.cat([gt_first, gt_last,  all_xs], dim=1)
                        # pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
                        # utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows+2)

                        self.save(milestone)

                pbar.update(1)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

        accelerator.print('training complete')