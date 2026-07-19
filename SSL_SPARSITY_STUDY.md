# SSL + Sparsity in RL — Study Plan & Handoff

Research question: how do self-supervised auxiliary tasks and sparsity methods affect
PPO performance, with a focus on **sample-efficient learning from pixels**.

Started 2026-07-19. Base: the PPO implementation in `RL/`.

---

## Status

**Step 1 (encoder refactor + seeding + logging + entropy fix): DONE, verified.**
Next: Step 2 (`SSLReplayBuffer` + `AuxTask` interface).

---

## Decisions already made (do not re-litigate)

| Decision | Choice |
|---|---|
| Stop-gradient | Test **both** as a factor: `detach_actor_encoder` True/False. Applies to SSL arms only. |
| SSL data source | **Separate replay buffer**, not PPO's on-policy rollouts. |
| Control condition | **`shared_encoder=True` with no SSL.** NOT the legacy unshared config. See "Critical" below. |
| Shared-path optimizer | RMSprop (matches original policy optimizer; Adam measured ~34% worse). |
| Sparsity level / SSL coeff | Treated as **independent variables** (report sensitivity curves), not nuisance hyperparameters to tune away. |

### Critical: baseline choice

Measured on custom CartPole, 20 iters, 2 seeds:

| config | score delta |
|---|---|
| unshared + rmsprop (legacy) | +26.47 |
| shared + rmsprop | +11.58 |
| shared + detach | +7.63 |

Encoder sharing costs real performance. If SSL is compared against the *legacy unshared*
baseline, a gain conflates "SSL improves representations" with "SSL recovers the sharing
penalty." Every condition and its control must share encoder topology.

Caveat: CartPole is the worst case for sharing (dense state, nothing to represent). **Re-measure
the sharing penalty on pixels before generalizing** — likely first task of step 2.

---

## Step 1: what was changed

### `RL/encoders.py` (new)
`MLPEncoder` / `CNNEncoder` exposing `output_dim`; `CNNEncoder` also exposes
`conv_out_shape` (SSL decoders need it). `ActorCritic` = encoder + policy head + value head,
flags `shared_encoder`, `detach_actor_encoder`, `feature_dim`.

Split is exact vs `common_networks.py`: unshared `ActorCritic` param counts match the
original two networks (MLP 9,220; CNN 678,471).

Known deviation: with a shared trunk the encoder gets orthogonal (policy-style) init; in the
unshared original the value net used PyTorch defaults. Documented in the file.

### `RL/PPO.py` (refactored)
- Entropy sign fixed: `policy_loss - ent_coef*H` (was `+`).
- Shared trunk + single optimizer with param groups (`lr` for encoder/policy, `value_lr` for value head).
- New kwargs: `shared_encoder`, `detach_actor_encoder`, `feature_dim`, `optimizer`, `vf_coef`, `seed`, `logger`.
- `encode(obs)` → shared features, for SSL heads and representation metrics.
- Legacy `models=(policy, value)` path still works (no shared trunk; SSL cannot attach).
- `train()` returns a stats dict.

**`_shared_backward()` — do not simplify to a single global clip.** The value gradient
measured 539× the policy gradient on the trunk (2.84 vs 0.0053); one global
`clip_grad_norm_` scaled everything by 0.176 and starved the actor (~11× less gradient than
legacy). Actor and critic paths are backpropagated and clipped **separately**, then
accumulated, then one step. SSL becomes a third independently-clipped path.

### `RL/experiment_utils.py` (new)
`set_global_seed()`; `RunLogger` → per-iteration `progress.csv` + `config.json` sidecar.
Columns inferred from the FIRST `log()` call — pass the full metric set from iteration 1
(use `None` for not-yet-available values) or later metrics get dropped.

### `RL/vec_env_handler.py`
`ParallelEnvManager(..., seed=)`; worker `i` seeded with `seed+i`. Workers are spawned
processes and do not inherit parent RNG state — without this nothing is reproducible.

### Verified
Param parity; `detach_actor_encoder` zeroes trunk policy grad (0.0 vs 1.81e-01); all three
configs train; same seed → bit-identical, different seed → diverges; CSV written.

Note: iteration-1 scores are identical across seeds because `policy_head.weight.div_(100)`
makes the initial policy uniform to within 0.004 (H = ln 3 exactly). Not a seeding bug.

### Entropy bug evidence
Isolated entropy term, from a collapsed policy: `+` sign drove H 0.8652 → 0.0149;
`−` sign restored 0.8652 → 1.0985 (= ln 3). The `+` was an exploration *penalty*.

Loose end: even fixed, `ent_coef` 0.0 vs 0.05 barely differs on CartPole (H 1.046 vs 1.048)
— policy gradient dominates. Mechanism is correct; may want retuning on harder envs.

---

## Remaining build order

2. `SSLReplayBuffer` + `AuxTask` interface + wire `detach_actor_encoder` through SSL.
3. Vertical slice: L1, magnitude pruning, autoencoder, contrastive (one per family).
4. Metrics module (effective rank, dormant ratio, measured sparsity, wall-clock).
5. Remaining methods → Tier 1 sweep → Tier 2 headline.

### Step 2 design (agreed)

`SSLReplayBuffer`, fed from `collect_rollouts`, own update cadence (n SSL grad steps per
rollout). Two requirements:
- **Store uint8, convert on sample.** 100k frames @ 3×64×64: 1.2 GB uint8 vs 4.9 GB float32.
  Do NOT copy `RolloutBuffer`'s all-float32 pattern.
- **Store transitions, not frames**: `(obs_t, obs_{t+k}, action_t, valid_mask)`, k configurable.
  Reconstructive methods use `obs_t` only; joint-embedding/JEPA need the pair; action-conditioned
  variants need the action. `valid_mask` handles episode boundaries — indexing `obs[t+1]`
  naively crosses episode and env boundaries.

Also: `RolloutBuffer.episode_starts` is allocated but never written (`buffers.py`) — dead now,
will be needed.

Interfaces:
```python
class AuxTask:
    def build(self, encoder_dim, obs_space): ...   # creates its own heads
    def loss(self, batch, encoder) -> Tensor: ...  # scalar
    def parameters(self): ...

class SparsityMethod:
    def on_init(self, modules): ...                # initial masks
    def regularizer(self, modules) -> Tensor: ...  # L1/L0 penalty, else 0
    def after_step(self, modules, step): ...       # SET/RigL regrow, pruning schedule
```
Both no-ops by default so baseline PPO stays on the same code path.

---

## Method grouping (13 items → shared implementations)

- **Weight sparsity, regularizer:** L1, L0 → just `regularizer()`.
- **Weight sparsity, masked:** magnitude pruning, SET, RigL — one masked-layer class, three
  regrow policies (none / random / gradient-magnitude).
- **Activation sparsity:** Top-K (k-WTA in forward pass, no masks). Different mechanism — group separately.
- **SSL reconstructive:** autoencoder → VQ-VAE (add codebook) → MAE (add masking); one decoder chain.
  Ladder networks are the outlier (lateral skips, invasive).
- **SSL joint-embedding:** contrastive (CURL-style), BYOL-style, JEPA — share augmentation +
  target-network + predictor scaffolding; differ in loss and whether prediction is in latent space.

---

## Metrics

Beyond return: AUC of learning curve (sample efficiency), steps-to-X%-of-baseline,
**measured** sparsity (SET/RigL drift), wall-clock per env step.

**Effective rank** (mechanism variable). Features Φ ∈ ℝ^(N×d), N ≈ 4096, singular values σ:
- `srank_δ`: smallest k with Σ_{i≤k}σ / Σσ ≥ 1−δ, δ=0.01. Interpretable, jumpy.
- **entropy-based** (default): p_i = σ_i/Σσ, report `exp(−Σ p_i log p_i)`. Continuous, no threshold.

Predictions: SSL should raise it (can't reconstruct from a collapsed representation);
sparsity could go either way — that's the interesting part.

Companion: **dormant neuron ratio** (fraction of units with normalized activation < τ ≈ 0.025).
Separates "unit was pruned" from "unit is alive but never fires" — matters for Top-K and dynamic sparsity.

Compute on a fixed held-out observation batch every K updates so runs are comparable.

**Statistics:** 5 seeds minimum, 10 preferred. Report IQM + stratified bootstrap CIs
(`rliable`), not mean ± std.

---

## Environments

Two-tier, required to make the sweeps affordable:

- **Tier 1 — MinAtar** (sweeps): 5 games, seconds-to-minutes/run, preserves pixel structure, ~100× cheaper than ALE.
- **Tier 2 — ALE** (headline): 3–5 games @ ~1–2M frames. `ALE/Breakout-v5` and
  `ALE/SpaceInvaders-v5` already wired in `gymnasium_main.py`. Pick *different* characteristics
  (dense reward, sparse reward, hard exploration), not 5 easy games.
- `cartpole_novelocity`: cheap partial-observability probe, not pixels.
- Procgen only if a generalization claim is wanted (separate scope).

Frame Tier 2 as relative comparison within PPO — **not** the Atari-100k benchmark; PPO is far
off that frontier and would be compared against DreamerV3/EfficientZero numbers.

Rough scope: Tier 1 ≈ 87 conditions × 3 games × 5 seeds ≈ 1300 runs (only sane because MinAtar
is cheap). Tier 2 ≈ 20 selected × 4 games × 5 seeds ≈ 400 runs. **Measure one run of each
end-to-end and multiply before committing.** If ALE > ~30 min/run, cut Tier 2 to 3 games.

Tune LR / SSL-updates-per-rollout on Tier 1 with 3 seeds, fix, transfer, and disclose it.
Tuning per-method-per-env is unaffordable; tuning only the baseline rigs the comparison.

---

## Environment notes

- Python 3.11, torch 2.11.0+cu130, CUDA available, 16 CPUs.
- `gymnasium` NOT installed — only custom envs in `RL/environments/` run right now.
  Needed for Tier 1/2. `numpy 2.4.3`, `opencv-python`, `matplotlib`, `scipy` present.
- **Any script spawning env workers needs `if __name__ == '__main__':`** (Windows spawn).
- Use `python -u` for scripts whose output is redirected, or nothing appears until exit.

Verification scripts live in the session scratchpad (not the repo): `entropy_gradient_check.py`,
`refactor_check.py`, `diagnose.py`, `grad_balance.py`, `optimizer_ablation.py`, `final_verify.py`.
They are regenerable from this document if lost.
