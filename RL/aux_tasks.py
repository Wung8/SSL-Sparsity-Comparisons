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


def _conv_input_shapes(encoder, observation_space):
    '''Spatial size entering each Conv2d, recorded by replaying the encoder stack.'''
    shapes = []
    with torch.no_grad():
        x = torch.zeros((1, *observation_space))
        for layer in encoder.cnn:
            if isinstance(layer, nn.Flatten):
                break
            if isinstance(layer, nn.Conv2d):
                shapes.append(tuple(x.shape[-2:]))
            x = layer(x)
    return shapes


def _output_padding(conv, cur_hw, tgt_hw):
    '''
    output_padding that makes a ConvTranspose2d invert `conv` back onto tgt spatial size,
    solved rather than hardcoded so it works for any input shape (see AutoencoderTask).
    '''
    k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    op = tuple(t - ((c - 1) * s - 2 * p + k) for c, t in zip(cur_hw, tgt_hw))
    assert all(0 <= o < max(s, 1) for o in op), (
        f"cannot invert conv: output_padding {op} out of range for stride {s}"
    )
    return op


def _build_mirror_decoder(encoder, observation_space, device):
    '''
    Decoder that mirrors the encoder's conv stack and maps the shared feature vector back
    to an observation. Shared by the whole reconstructive family (autoencoder, VQ-VAE,
    masked VQ-VAE) so "one decoder chain" is literally one function, per the study plan.
    '''
    if not hasattr(encoder, "conv_out_shape"):
        raise ValueError("reconstructive SSL needs a CNNEncoder (conv_out_shape missing)")

    convs = [l for l in encoder.cnn if isinstance(l, nn.Conv2d)]
    shapes = _conv_input_shapes(encoder, observation_space)

    c_out, h_out, w_out = encoder.conv_out_shape
    layers = [
        nn.Linear(encoder.output_dim, encoder.n_flatten),
        nn.Mish(),
        nn.Unflatten(-1, (c_out, h_out, w_out)),
    ]

    cur = (h_out, w_out)
    for i, conv in enumerate(reversed(convs)):
        tgt = shapes[len(convs) - 1 - i]
        op = _output_padding(conv, cur, tgt)
        last = i == len(convs) - 1
        layers.append(nn.ConvTranspose2d(
            conv.out_channels, conv.in_channels,
            kernel_size=conv.kernel_size[0], stride=conv.stride[0],
            padding=conv.padding[0], output_padding=op,
        ))
        if not last:
            layers.append(nn.Mish())
        cur = tgt

    return nn.Sequential(*layers).to(device)


def random_patch_mask(obs, patch=8, ratio=0.5):
    '''
    MAE-style masking for a CNN: zero out a fraction of square patches. A single-vector
    CNN encoder has no per-patch tokens, so this is the tractable analog of MAE -- corrupt
    the input by patches, reconstruct the ORIGINAL. Returns (masked_obs, keep_mask).
    '''
    b, c, h, w = obs.shape
    gh, gw = h // patch, w // patch
    keep = (torch.rand(b, 1, gh, gw, device=obs.device) > ratio).float()
    keep = keep.repeat_interleave(patch, 2).repeat_interleave(patch, 3)
    # pad to full size if h/w not divisible by patch (kept regions default to visible)
    if keep.shape[-2:] != (h, w):
        full = torch.ones(b, 1, h, w, device=obs.device)
        full[..., : keep.shape[-2], : keep.shape[-1]] = keep
        keep = full
    return obs * keep, keep


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
        self.decoder = _build_mirror_decoder(encoder, observation_space, device)

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


class VectorQuantizer(nn.Module):
    '''
    Product (grouped) vector quantization of the shared feature vector.

    A single 256-d representation quantized against one codebook would collapse the whole
    state to one of K discrete symbols -- far too coarse. So the vector is split into
    `n_groups` sub-vectors, each quantized against its own codebook, giving K**n_groups
    effective codes. This is the standard product-VQ trick for quantizing a global vector
    rather than a spatial feature grid.

    Straight-through estimator passes the reconstruction gradient to the encoder; the VQ
    loss (codebook + commitment) trains the codebook and pulls encoder outputs toward it.
    '''

    def __init__(self, dim, n_codes=512, n_groups=8, commitment=0.25):
        super().__init__()
        assert dim % n_groups == 0, f"feature dim {dim} not divisible by n_groups {n_groups}"
        self.n_groups = n_groups
        self.group_dim = dim // n_groups
        self.n_codes = n_codes
        self.commitment = commitment
        self.codebook = nn.Parameter(torch.randn(n_groups, n_codes, self.group_dim) * 0.1)

    def forward(self, z):
        b = z.shape[0]
        zg = z.view(b, self.n_groups, self.group_dim)          # (B, G, d)
        cb = self.codebook                                     # (G, K, d)

        # squared distances (B, G, K): |z|^2 - 2 z.e + |e|^2
        z2 = (zg ** 2).sum(-1, keepdim=True)
        e2 = (cb ** 2).sum(-1).unsqueeze(0)
        ze = torch.einsum("bgd,gkd->bgk", zg, cb)
        idx = (z2 - 2 * ze + e2).argmin(-1)                    # (B, G)

        z_q = torch.stack([cb[g][idx[:, g]] for g in range(self.n_groups)], dim=1)  # (B,G,d)

        codebook_loss = F.mse_loss(z_q, zg.detach())
        commit_loss = F.mse_loss(zg, z_q.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss

        z_q_st = zg + (z_q - zg).detach()                      # straight-through
        with torch.no_grad():
            usage = F.one_hot(idx, self.n_codes).float().mean(dim=(0, 1))
            perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())
        return z_q_st.reshape(b, -1), vq_loss, perplexity


class VQVAETask(AuxTask):
    '''
    Reconstructive SSL with a discrete bottleneck: autoencoder + a product-VQ codebook
    between encoder and decoder. Same decoder chain as AutoencoderTask; the only addition
    is quantization of the shared features before decoding.
    '''

    name = "vqvae"
    k_step = 0

    def __init__(self, n_codes=512, n_groups=8, commitment=0.25):
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.commitment = commitment
        self.decoder = None
        self.vq = None
        self._last_perplexity = None

    def build(self, encoder, observation_space, action_space, device):
        self.decoder = _build_mirror_decoder(encoder, observation_space, device)
        self.vq = VectorQuantizer(encoder.output_dim, self.n_codes,
                                  self.n_groups, self.commitment).to(device)

    def loss(self, batch, encoder):
        z = encoder(batch.obs)
        z_q, vq_loss, perplexity = self.vq(z)
        self._last_perplexity = perplexity.item()
        recon = self.decoder(z_q)
        return F.mse_loss(recon, batch.obs) + vq_loss

    def parameters(self):
        return list(self.decoder.parameters()) + list(self.vq.parameters())

    def modules(self):
        return [m for m in (self.decoder, self.vq) if m is not None]

    def config(self):
        return {"aux_task": self.name, "n_codes": self.n_codes,
                "n_groups": self.n_groups, "commitment": self.commitment}


class MaskedVQVAETask(VQVAETask):
    '''
    VQ-VAE with MAE-style input masking: encode a patch-masked observation, quantize,
    reconstruct the ORIGINAL (unmasked) frame. The masking forces the representation to
    fill in occluded regions rather than copy, which is the point of masked autoencoding.

    Reconstructing the original rather than only the masked patches keeps this a single
    global-decoder objective -- a per-patch loss would need per-patch tokens the CNN
    encoder does not produce.
    '''

    name = "masked_vqvae"
    k_step = 0

    def __init__(self, n_codes=512, n_groups=8, commitment=0.25, mask_ratio=0.5, patch=8):
        super().__init__(n_codes, n_groups, commitment)
        self.mask_ratio = mask_ratio
        self.patch = patch

    def loss(self, batch, encoder):
        masked, _ = random_patch_mask(batch.obs, self.patch, self.mask_ratio)
        z = encoder(masked)
        z_q, vq_loss, perplexity = self.vq(z)
        self._last_perplexity = perplexity.item()
        recon = self.decoder(z_q)
        return F.mse_loss(recon, batch.obs) + vq_loss

    def config(self):
        c = super().config()
        c.update(aux_task=self.name, mask_ratio=self.mask_ratio, patch=self.patch)
        return c


class JEPATask(AuxTask):
    '''
    Joint-Embedding Predictive Architecture: predict the EMA-target embedding of one view
    from a predictor on the online embedding of another view, in LATENT space, with no
    negatives (unlike contrastive) and no pixel decoder (unlike the reconstructive chain).

    Reuses the ContrastiveTask scaffolding -- an EMA target encoder + two augmented views
    -- but the loss is a latent regression, and an asymmetric predictor sits on the online
    branch. Collapse to a constant is prevented by the EMA target + predictor asymmetry,
    the same mechanism as BYOL; there are no negatives holding the space apart, so the
    effective-rank metric is the thing to watch on this arm.

    I-JEPA's spatial mask-and-predict does not map onto a global-pooled CNN encoder (no
    per-patch tokens), so this is the tractable single-vector JEPA.
    '''

    name = "jepa"
    k_step = 0

    def __init__(self, momentum=0.05, hidden=512, pad=4):
        self.momentum = momentum
        self.hidden = hidden
        self.pad = pad
        self.target_encoder = None
        self.predictor = None
        self._live_encoder = None

    def build(self, encoder, observation_space, action_space, device):
        self.target_encoder = copy.deepcopy(encoder).to(device)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        dim = encoder.output_dim
        self.predictor = nn.Sequential(
            nn.Linear(dim, self.hidden), nn.Mish(), nn.Linear(self.hidden, dim),
        ).to(device)
        self._live_encoder = encoder

    def loss(self, batch, encoder):
        online = self.predictor(encoder(random_shift(batch.obs, self.pad)))
        with torch.no_grad():
            target = self.target_encoder(random_shift(batch.obs, self.pad))
        # BYOL-style normalised MSE: 2 - 2*cos, symmetric in scale, bounded
        online = F.normalize(online, dim=-1)
        target = F.normalize(target, dim=-1)
        return (2 - 2 * (online * target).sum(-1)).mean()

    def after_step(self):
        if self.target_encoder is None:
            return
        with torch.no_grad():
            for tp, lp in zip(self.target_encoder.parameters(),
                              self._live_encoder.parameters()):
                tp.mul_(1 - self.momentum).add_(self.momentum * lp.detach())

    def parameters(self):
        return list(self.predictor.parameters())

    def modules(self):
        return [m for m in (self.target_encoder, self.predictor) if m is not None]

    def config(self):
        return {"aux_task": self.name, "momentum": self.momentum, "hidden": self.hidden,
                "pad": self.pad}


class LadderDecoder(nn.Module):
    '''
    Top-down denoising decoder with lateral connections into every encoder layer -- the
    defining structure of a ladder network. Built from the encoder's own conv geometry so
    it mirrors any input shape.

    Each level combines the (noisy) lateral activation with the top-down reconstruction
    from the level above, and is trained to match that level's CLEAN activation. This is
    the layer-wise reconstruction that distinguishes ladder nets from a plain autoencoder,
    which only reconstructs the input.
    '''

    def __init__(self, encoder, observation_space):
        super().__init__()
        convs = [l for l in encoder.cnn if isinstance(l, nn.Conv2d)]
        shapes = _conv_input_shapes(encoder, observation_space)  # input spatial per conv
        c_out, h_out, w_out = encoder.conv_out_shape
        dim = encoder.output_dim

        # denoise the top representation vector, then project it into conv space
        self.top = nn.Sequential(nn.Linear(dim, dim), nn.Mish())
        self.to_conv = nn.Sequential(
            nn.Linear(dim, encoder.n_flatten), nn.Mish(),
            nn.Unflatten(-1, (c_out, h_out, w_out)),
        )

        # per-conv combinator (fuse lateral + top-down) and upsampler (to the level below)
        self.combinators = nn.ModuleList()
        self.ups = nn.ModuleList()
        n = len(convs)
        cur = (h_out, w_out)
        for i, conv in enumerate(reversed(convs)):          # top conv -> bottom conv
            ch = conv.out_channels
            self.combinators.append(nn.Sequential(
                nn.Conv2d(2 * ch, ch, kernel_size=3, padding=1), nn.Mish()))
            tgt = shapes[n - 1 - i]
            if i < n - 1:
                op = _output_padding(conv, cur, tgt)
                self.ups.append(nn.ConvTranspose2d(
                    ch, conv.in_channels, kernel_size=conv.kernel_size[0],
                    stride=conv.stride[0], padding=conv.padding[0], output_padding=op))
            else:
                self.ups.append(None)                       # bottom level: no further down
            cur = tgt

    def forward(self, noisy_acts):
        # noisy_acts = [conv1, conv2, ..., convL, top_vector], bottom -> top
        conv_acts = noisy_acts[:-1]
        top = noisy_acts[-1]

        recon_top = self.top(top)
        u = self.to_conv(recon_top)                          # matches the top conv act
        recon_convs = [None] * len(conv_acts)
        for j, i in enumerate(reversed(range(len(conv_acts)))):   # top conv down
            fused = self.combinators[j](torch.cat([conv_acts[i], u], dim=1))
            recon_convs[i] = fused
            if self.ups[j] is not None:
                u = self.ups[j](fused)
        return recon_top, recon_convs


class LadderTask(AuxTask):
    '''
    Ladder network: the invasive outlier of the SSL set. Unlike every other task here it
    needs the encoder's INTERMEDIATE activations, not just its final embedding, so it
    forwards through encoder.cnn / encoder.fc by hand to collect them (no hooks, no
    encoder edits).

    Objective: run a clean pass (targets, detached) and a noise-corrupted pass (gradient
    path), then a top-down decoder with lateral connections reconstructs the clean
    activation at every layer from the noisy one. Layer-wise denoising is what makes this
    a ladder rather than an autoencoder.
    '''

    name = "ladder"
    k_step = 0

    def __init__(self, noise_std=0.3, layer_weight=0.1):
        self.noise_std = noise_std
        self.layer_weight = layer_weight
        self.decoder = None

    def build(self, encoder, observation_space, action_space, device):
        if not hasattr(encoder, "conv_out_shape"):
            raise ValueError("LadderTask needs a CNNEncoder")
        self.decoder = LadderDecoder(encoder, observation_space).to(device)

    @staticmethod
    def _collect(encoder, x):
        '''Post-Mish activations at each conv block plus the final feature vector.'''
        acts = []
        h = x
        for layer in encoder.cnn:
            h = layer(h)
            if isinstance(layer, nn.Mish):
                acts.append(h)
        for layer in encoder.fc:            # Linear then Mish
            h = layer(h)
        acts.append(h)                      # final representation vector
        return acts

    @staticmethod
    def _standardize(a):
        '''
        Per-sample zero-mean unit-variance over the feature dims.

        Ladder nets denoise toward the encoder's OWN clean activations, which lets the
        encoder drive the loss down by simply inflating activation magnitude without bound
        -- measured: top-vector norm ran from 1 to >1000 and the loss diverged. The
        original ladder controls this with batch-norm at every layer; standardising both
        sides here is the same fix, making the objective compare PATTERN, not scale.
        '''
        dims = tuple(range(1, a.dim()))
        mean = a.mean(dim=dims, keepdim=True)
        std = a.std(dim=dims, keepdim=True)
        return (a - mean) / (std + 1e-5)

    def loss(self, batch, encoder):
        with torch.no_grad():
            clean = [self._standardize(a).detach() for a in self._collect(encoder, batch.obs)]
        noisy_in = batch.obs + self.noise_std * torch.randn_like(batch.obs)
        noisy = self._collect(encoder, noisy_in)            # gradient path

        recon_top, recon_convs = self.decoder(noisy)
        loss = F.mse_loss(self._standardize(recon_top), clean[-1])   # top representation
        for rec, tgt in zip(recon_convs, clean[:-1]):       # every conv layer
            loss = loss + self.layer_weight * F.mse_loss(self._standardize(rec), tgt)
        return loss

    def parameters(self):
        return list(self.decoder.parameters())

    def modules(self):
        return [self.decoder] if self.decoder is not None else []

    def config(self):
        return {"aux_task": self.name, "noise_std": self.noise_std,
                "layer_weight": self.layer_weight}


AUX_TASKS = {
    "none": NoAuxTask,
    "autoencoder": AutoencoderTask,
    "contrastive": ContrastiveTask,
    "vqvae": VQVAETask,
    "masked_vqvae": MaskedVQVAETask,
    "jepa": JEPATask,
    "ladder": LadderTask,
}


def make_aux_task(name, **kwargs):
    if name is None or name == "none":
        return NoAuxTask()
    return AUX_TASKS[name](**kwargs)
