import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

'''
Self-supervised auxiliary tasks that attach to the shared encoder.

Contract (mirrors SparsityMethod in RL/sparsity.py):

    build(encoder, observation_space, action_space, device)  -- create own heads
    loss(batch, encoder) -> scalar Tensor                     -- SSL objective
    parameters() -> list                                      -- head params for the optimizer
    after_step()                                              -- momentum/EMA bookkeeping

NoAuxTask is the default and is a strict no-op, so baseline PPO stays on exactly the same
code path as an SSL run with the task removed.

Note on detach_actor_encoder: it is NOT handled here. The SSL loss always reaches the
encoder via encoder(...); the flag only controls whether the *policy* gradient does, and
that is enforced in ActorCritic.forward. detach=True + an SSL task is the CURL/SAC-AE
regime (encoder shaped by critic + SSL, not by the actor); detach=True with NO task
starves the encoder down to the value signal alone and is not a meaningful condition.
'''


def random_shift(imgs, pad=4):
    '''
    DrQ-style random shift: replicate-pad then crop back to the original size. Cheaper
    and better behaved at 64x64 than CURL's 100->84 random crop, which assumes a larger
    source frame than these observations have.
    '''
    n, c, h, w = imgs.shape
    padded = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    top = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
    left = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
    rows = torch.arange(h, device=imgs.device)
    cols = torch.arange(w, device=imgs.device)
    idx_r = top[:, None] + rows[None, :]
    idx_c = left[:, None] + cols[None, :]
    batch = torch.arange(n, device=imgs.device)[:, None, None]
    return padded[batch, :, idx_r[:, :, None], idx_c[:, None, :]].permute(0, 3, 1, 2)


class AuxTask:
    name = "none"
    # k-step gap the task needs from the replay buffer. 0 = single frames.
    k_step = 0

    def build(self, encoder, observation_space, action_space, device):
        pass

    def loss(self, batch, encoder):
        return None

    def parameters(self):
        return []

    def after_step(self):
        pass

    def modules(self):
        return []

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)
        return self

    def config(self):
        return {"aux_task": self.name}


class NoAuxTask(AuxTask):
    '''Explicit null object so the baseline needs no branching at the call site.'''
    name = "none"


class AutoencoderTask(AuxTask):
    '''
    Reconstructive SSL: decode obs_t back from the shared features.

    The decoder mirrors the encoder's conv stack exactly, with output_padding solved per
    layer so the reconstruction lands on the original spatial size for any input shape
    (hardcoding it would only work for 64x64).

    This is the base of the reconstructive chain -- VQ-VAE adds a codebook between
    encoder and decoder, MAE adds input masking; both reuse this decoder.
    '''

    name = "autoencoder"
    k_step = 0

    def __init__(self, latent_noise=0.0):
        self.latent_noise = latent_noise
        self.decoder = None

    def build(self, encoder, observation_space, action_space, device):
        if not hasattr(encoder, "conv_out_shape"):
            raise ValueError("AutoencoderTask needs a CNNEncoder (conv_out_shape missing)")

        convs = [l for l in encoder.cnn if isinstance(l, nn.Conv2d)]

        # replay the encoder to record the spatial size entering each conv layer
        shapes = []
        with torch.no_grad():
            x = torch.zeros((1, *observation_space))
            for layer in encoder.cnn:
                if isinstance(layer, nn.Flatten):
                    break
                if isinstance(layer, nn.Conv2d):
                    shapes.append(tuple(x.shape[-2:]))
                x = layer(x)

        c_out, h_out, w_out = encoder.conv_out_shape
        layers = [
            nn.Linear(encoder.output_dim, encoder.n_flatten),
            nn.Mish(),
            nn.Unflatten(-1, (c_out, h_out, w_out)),
        ]

        cur_h, cur_w = h_out, w_out
        for i, conv in enumerate(reversed(convs)):
            tgt_h, tgt_w = shapes[len(convs) - 1 - i]
            k = conv.kernel_size[0]
            s = conv.stride[0]
            p = conv.padding[0]
            op_h = tgt_h - ((cur_h - 1) * s - 2 * p + k)
            op_w = tgt_w - ((cur_w - 1) * s - 2 * p + k)
            assert 0 <= op_h < max(s, 1) and 0 <= op_w < max(s, 1), (
                f"cannot invert conv {i}: output_padding ({op_h},{op_w}) out of range for stride {s}"
            )
            last = i == len(convs) - 1
            layers.append(nn.ConvTranspose2d(
                conv.out_channels, conv.in_channels,
                kernel_size=k, stride=s, padding=p,
                output_padding=(op_h, op_w),
            ))
            if not last:
                layers.append(nn.Mish())
            cur_h, cur_w = tgt_h, tgt_w

        self.decoder = nn.Sequential(*layers).to(device)

    def loss(self, batch, encoder):
        z = encoder(batch.obs)
        if self.latent_noise > 0:
            z = z + self.latent_noise * torch.randn_like(z)
        recon = self.decoder(z)
        return F.mse_loss(recon, batch.obs)

    def parameters(self):
        return list(self.decoder.parameters())

    def modules(self):
        return [self.decoder] if self.decoder is not None else []

    def config(self):
        return {"aux_task": self.name, "latent_noise": self.latent_noise}


class ContrastiveTask(AuxTask):
    '''
    CURL-style instance contrastive learning.

    Two random-shift views of obs_t. The query view goes through the live shared encoder,
    the key view through an EMA copy. Similarity is the bilinear q^T W k of CURL, and the
    InfoNCE target is the matching index along the batch.

    The EMA copy + projection scaffolding here is what BYOL-style and JEPA reuse later;
    those differ in the loss and in whether prediction happens in latent space, not in
    this machinery.

    Gradients reach the shared encoder through the query path only -- the key encoder is
    updated by EMA and is explicitly excluded from autograd, which is what stops the
    representation collapsing to a constant.
    '''

    name = "contrastive"
    k_step = 0

    def __init__(self, momentum=0.05, temperature=1.0, pad=4):
        self.momentum = momentum          # EMA rate toward the live encoder
        # temperature defaults to 1.0 (i.e. off) because CURL's bilinear W already learns
        # its own scale -- a second temperature just multiplies it. That matters more
        # than it looks: these logits are NOT normalised, so they grow with ||feat||^2 as
        # training proceeds, and a 0.1 temperature scales that growth by another 10x.
        # Kept as a parameter so it can be swept, but do not treat it as free.
        self.temperature = temperature
        self.pad = pad
        self.key_encoder = None
        self.W = None
        self._live_encoder = None

    def build(self, encoder, observation_space, action_space, device):
        self.key_encoder = copy.deepcopy(encoder).to(device)
        for p in self.key_encoder.parameters():
            p.requires_grad_(False)
        dim = encoder.output_dim
        self.W = nn.Parameter(torch.rand(dim, dim, device=device) * 0.01)
        self._live_encoder = encoder

    def loss(self, batch, encoder):
        obs = batch.obs
        query_view = random_shift(obs, self.pad)
        key_view = random_shift(obs, self.pad)

        q = encoder(query_view)
        with torch.no_grad():
            k = self.key_encoder(key_view)

        # CURL bilinear similarity: (B,d) @ (d,d) @ (d,B) -> (B,B)
        logits = q @ (self.W @ k.T)
        # subtracting the row max is the standard InfoNCE stabiliser
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        logits = logits / self.temperature

        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def after_step(self):
        '''EMA the key encoder toward the live one. Called after each SSL optimizer step.'''
        if self.key_encoder is None:
            return
        with torch.no_grad():
            for kp, qp in zip(self.key_encoder.parameters(), self._live_encoder.parameters()):
                kp.mul_(1 - self.momentum).add_(self.momentum * qp.detach())

    def parameters(self):
        return [self.W]

    def modules(self):
        return [self.key_encoder] if self.key_encoder is not None else []

    def to(self, device):
        super().to(device)
        if self.W is not None:
            self.W.data = self.W.data.to(device)
        return self

    def config(self):
        return {
            "aux_task": self.name,
            "momentum": self.momentum,
            "temperature": self.temperature,
            "pad": self.pad,
        }


AUX_TASKS = {
    "none": NoAuxTask,
    "autoencoder": AutoencoderTask,
    "contrastive": ContrastiveTask,
}


def make_aux_task(name, **kwargs):
    if name is None or name == "none":
        return NoAuxTask()
    return AUX_TASKS[name](**kwargs)
