import numpy as np
import torch
from torch import nn

'''
Sparsity methods applied to the actor-critic.

Contract (mirrors AuxTask in RL/aux_tasks.py):

    on_init(modules)                  -- initial masks, if any
    regularizer(modules) -> Tensor    -- L1/L0 penalty, else None
    after_step(modules, step)         -- pruning schedule, SET/RigL regrow
    measured_sparsity(modules)        -- ACTUAL zero fraction, not the target

NoSparsity is the default and a strict no-op.

measured_sparsity exists because the target and the reality diverge: dynamic methods
(SET, RigL) drift as they prune and regrow, and L1 pushes weights toward zero without
ever making them exactly zero. Reporting the requested level instead of the achieved one
is how sparsity papers end up incomparable.

Which layers: Conv2d and Linear *weights* in the encoder (and value_encoder when
unshared). Biases are excluded -- they are a rounding error in parameter count and
zeroing them is a different intervention. The policy and value output heads are excluded
by default: they are ~770 of 341k parameters here, and pruning a 256->2 output layer is
a qualitatively different operation from pruning a trunk. Set include_heads=True to
override.
'''


def _prunable(modules, include_heads=False):
    '''Yields (module, param_name, param) for every prunable weight tensor.'''
    out = []
    for name, module in modules.items():
        if not include_heads and name in ("policy_head", "value_head"):
            continue
        if module is None:
            continue
        for m in module.modules() if isinstance(module, nn.Module) else []:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                out.append((m, "weight", m.weight))
    return out


class SparsityMethod:
    name = "none"

    def on_init(self, modules):
        pass

    def regularizer(self, modules):
        return None

    def after_step(self, modules, step):
        pass

    def measured_sparsity(self, modules):
        params = [p for _, _, p in _prunable(modules, getattr(self, "include_heads", False))]
        if not params:
            return 0.0
        total = sum(p.numel() for p in params)
        zeros = sum((p == 0).sum().item() for p in params)
        return zeros / total

    def config(self):
        return {"sparsity": self.name}


class NoSparsity(SparsityMethod):
    name = "none"


class L1Regularization(SparsityMethod):
    '''
    Soft sparsity: add coef * sum|w| to the loss.

    Does NOT produce exact zeros -- measured_sparsity will report ~0 even at a coef that
    visibly shrinks the weights. That is the honest number; report the weight-magnitude
    distribution alongside it rather than thresholding to manufacture a sparsity figure.
    '''

    name = "l1"

    def __init__(self, coef=1e-4, include_heads=False):
        self.coef = coef
        self.include_heads = include_heads

    def regularizer(self, modules):
        params = [p for _, _, p in _prunable(modules, self.include_heads)]
        if not params:
            return None
        return self.coef * sum(p.abs().sum() for p in params)

    def config(self):
        return {"sparsity": self.name, "l1_coef": self.coef, "include_heads": self.include_heads}


class MagnitudePruning(SparsityMethod):
    '''
    Gradual magnitude pruning, Zhu & Gupta cubic schedule:

        s_t = s_f * (1 - (1 - progress)^3)

    so most of the pruning happens early and the tail is gentle, letting the network
    recover between steps. Masks are recomputed every `frequency` steps between
    start_step and end_step, then held fixed.

    Global ranking (one threshold across all prunable layers) rather than per-layer, so
    layers that genuinely need capacity keep it. The cost is that a small layer can be
    pruned away entirely at high sparsity -- worth watching in the dormant-neuron metric.

    Masks are re-applied after every optimizer step, so a pruned weight stays at zero
    even though the optimizer keeps momentum for it.
    '''

    name = "magnitude_pruning"

    def __init__(self, target_sparsity=0.9, start_step=0, end_step=2000,
                 frequency=100, include_heads=False):
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
        self.include_heads = include_heads
        self.masks = {}
        self.current_sparsity = 0.0

    def _schedule(self, step):
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return self.target_sparsity
        span = max(self.end_step - self.start_step, 1)
        progress = (step - self.start_step) / span
        return self.target_sparsity * (1 - (1 - progress) ** 3)

    def on_init(self, modules):
        self.masks = {}
        for m, pname, p in _prunable(modules, self.include_heads):
            self.masks[id(p)] = torch.ones_like(p)

    def _recompute_masks(self, modules, sparsity):
        params = [p for _, _, p in _prunable(modules, self.include_heads)]
        if not params or sparsity <= 0:
            return
        # global threshold over the magnitudes of all currently-live weights
        flat = torch.cat([p.detach().abs().flatten() for p in params])
        k = int(sparsity * flat.numel())
        if k <= 0:
            return
        threshold = torch.kthvalue(flat, k).values
        for p in params:
            self.masks[id(p)] = (p.detach().abs() > threshold).float()

    def after_step(self, modules, step):
        target = self._schedule(step)

        in_window = self.start_step <= step <= self.end_step
        due = (step - self.start_step) % self.frequency == 0
        if in_window and due and target > 0:
            self._recompute_masks(modules, target)
            self.current_sparsity = target

        # re-apply every step: the optimizer would otherwise resurrect pruned weights
        # via momentum and weight decay
        if self.masks:
            with torch.no_grad():
                for _, _, p in _prunable(modules, self.include_heads):
                    mask = self.masks.get(id(p))
                    if mask is None:
                        continue
                    # PPO.set_training_mode moves the network cpu<->cuda every iteration,
                    # so a mask built on one device will not match the param on the next
                    # call. Follow the param and cache the moved copy.
                    if mask.device != p.device:
                        mask = mask.to(p.device)
                        self.masks[id(p)] = mask
                    p.mul_(mask)

    def config(self):
        return {
            "sparsity": self.name,
            "target_sparsity": self.target_sparsity,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "frequency": self.frequency,
            "include_heads": self.include_heads,
        }


SPARSITY_METHODS = {
    "none": NoSparsity,
    "l1": L1Regularization,
    "magnitude_pruning": MagnitudePruning,
}


def make_sparsity(name, **kwargs):
    if name is None or name == "none":
        return NoSparsity()
    return SPARSITY_METHODS[name](**kwargs)
