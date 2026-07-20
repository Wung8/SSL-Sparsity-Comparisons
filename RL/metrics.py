import numpy as np
import torch

'''
Representation-health metrics, computed on a FIXED held-out observation batch so numbers
are comparable across runs and across training time. Re-sampling the batch each time
would mix representation drift with data drift.

These are the mechanism variables of the study: return says whether a method worked,
these say why. The prediction is that SSL raises effective rank (you cannot reconstruct
or discriminate from a collapsed representation) while sparsity could push either way --
that is the part worth measuring rather than assuming.
'''


def effective_rank_entropy(features):
    '''
    Entropy-based effective rank: p_i = sigma_i / sum(sigma), report exp(H(p)).

    Default over srank_delta because it is continuous and threshold-free. Ranges from 1
    (rank-1 / fully collapsed) to d (all directions equally used).
    '''
    sv = _singular_values(features)
    if sv is None:
        return float("nan")
    total = sv.sum()
    if total <= 0:
        return float("nan")
    p = sv / total
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def srank_delta(features, delta=0.01):
    '''
    Smallest k whose top-k singular values capture (1 - delta) of the spectral mass.

    Reported alongside the entropy version because it is the more common definition in
    the literature, but it is jumpy near the threshold -- do not read small changes.
    '''
    sv = _singular_values(features)
    if sv is None:
        return float("nan")
    total = sv.sum()
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(sv) / total
    return int(np.searchsorted(cumulative, 1.0 - delta) + 1)


def _singular_values(features):
    if isinstance(features, torch.Tensor):
        features = features.detach().float().cpu().numpy()
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or min(features.shape) == 0:
        return None
    # centre: an uncentred mean direction inflates the leading singular value and makes
    # every representation look lower-rank than it is
    features = features - features.mean(axis=0, keepdims=True)
    try:
        return np.linalg.svd(features, compute_uv=False)
    except np.linalg.LinAlgError:
        return None


def dormant_ratio(activations, tau=0.025):
    '''
    Fraction of units whose normalised mean absolute activation is <= tau
    (Sokar et al. dormant-neuron ratio).

    Score for unit i in a layer of H units:
        s_i = E_x|h_i(x)| / ( (1/H) * sum_j E_x|h_j(x)| )

    The normalisation is what makes tau comparable across layers and across training --
    a raw magnitude threshold would just track the overall activation scale.

    Companion to measured sparsity: it separates "this unit was pruned away" from "this
    unit still exists but never fires", which is the distinction that matters for Top-K
    and for dynamic sparsity methods.
    '''
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().float().cpu().numpy()
    activations = np.asarray(activations, dtype=np.float64)
    if activations.ndim != 2 or activations.shape[1] == 0:
        return float("nan")

    mean_abs = np.abs(activations).mean(axis=0)
    denom = mean_abs.mean()
    if denom <= 0:
        return 1.0
    scores = mean_abs / denom
    return float((scores <= tau).mean())


def weight_sparsity(modules, include_heads=False):
    '''Measured zero fraction over prunable weights. Duplicates SparsityMethod.
    measured_sparsity so it can be reported for arms with no sparsity method attached.'''
    from RL.sparsity import _prunable
    params = [p for _, _, p in _prunable(modules, include_heads)]
    if not params:
        return 0.0
    total = sum(p.numel() for p in params)
    zeros = sum((p == 0).sum().item() for p in params)
    return zeros / total


@torch.no_grad()
def representation_report(encoder, obs_batch, modules=None, tau=0.025, delta=0.01):
    '''
    All representation metrics from one forward pass on the held-out batch.

    obs_batch: float32 tensor already on the encoder's device.
    '''
    was_training = getattr(encoder, "training", False)
    encoder.eval()
    features = encoder(obs_batch)
    encoder.train(was_training)

    report = {
        "eff_rank": effective_rank_entropy(features),
        "srank": srank_delta(features, delta),
        "dormant_ratio": dormant_ratio(features, tau),
        "feature_dim": int(features.shape[1]),
    }
    if modules is not None:
        report["measured_sparsity"] = weight_sparsity(modules)
    return report


class HeldOutBatch:
    '''
    Collects a fixed observation batch once, then serves it unchanged for the rest of
    the run. N ~ 4096 per the study plan: comfortably above feature_dim=256 so the
    spectrum is well estimated, and small enough that the SVD is free next to training.
    '''

    def __init__(self, n=4096, device="cpu"):
        self.n = n
        self.device = device
        self.obs = None

    def is_ready(self):
        return self.obs is not None

    def fill_from(self, buffer):
        '''Draw from an SSLReplayBuffer once it holds enough frames.'''
        if self.obs is not None:
            return True
        if buffer.n_frames() < self.n:
            return False
        batch = buffer.sample(self.n, k=0, device=self.device)
        self.obs = batch.obs
        return True
