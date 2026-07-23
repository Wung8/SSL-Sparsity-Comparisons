# SSL + Sparsity in RL — Study Plan & Handoff

Research question: how do self-supervised auxiliary tasks and sparsity methods affect
PPO performance, with a focus on **sample-efficient learning from pixels**.

Started 2026-07-19. Base: the PPO implementation in `RL/`.

---

## Status

**Steps 1–4: DONE, verified.** Step 1 (encoder refactor + seeding + logging + entropy
fix); Step 2 (`SSLReplayBuffer` + `AuxTask`/`SparsityMethod` interfaces); Step 3 (vertical
slice: L1, magnitude pruning, autoencoder, contrastive); Step 4 (metrics module).

45 unit tests + 30 integration tests pass. The baseline path is proven **bit-identical**
to pre-refactor `PPO.py` under cudnn-deterministic (see "Reproducibility" below).

**Step 5 IN PROGRESS:** ALE/Breakout sweep, 17 conditions × 5 seeds = 85 runs @ 4M frames,
~59h. Resumable — any cell with a `summary.json` is skipped.

**The sharing-penalty question is settled on pixels. See below — the CartPole result did
NOT transfer, and the baseline decision needs restating because of it.**

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

### RESOLVED: the sharing penalty on pixels (FlappyBirdImg, 3 seeds, 1.44M frames)

Measured 2026-07-19. Arms are `shared_encoder=False` (verified param-identical to the
legacy two-network pair), `shared_encoder=True`, and `+detach_actor_encoder`.

| arm | final10 | AUC | params | wall |
|---|---|---|---|---|
| unshared (legacy) | 10.77 ± 1.02 | **8.18 ± 0.43** | 681,539 | 20.9 min |
| shared | **11.28 ± 1.07** | 5.70 ± 0.53 | 341,155 | 16.3 min |
| detach (no SSL) | 1.71 ± 0.36 | 0.51 ± 0.43 | 341,155 | 16.4 min |

**The CartPole finding does not transfer.** On CartPole sharing cost final performance
outright (−56%). On pixels there is **no final-performance penalty at all** — shared is
nominally +4.7% ahead, ranks swap across seeds, ranges overlap almost completely — on
**half the parameters and 21% less wall-clock**. What remains is a *sample-efficiency*
penalty: −30.3% AUC, with `shared` [5.11, 6.39] and `unshared` [7.58, 8.54] fully
non-overlapping across 3/3 seeds.

The learning curves show why the two metrics disagree. `unshared` takes off ~10 iterations
earlier and peaks higher (12.73 at iter 45) **then degrades 19% to 10.37 by iter 60**;
`shared` takes off later, plateaus ~11.1–11.8, and holds — overtaking `unshared` at the
end. So this is not "same destination, different speed"; it is an early-but-unstable curve
against a late-but-stable one. Figure: `fig1_sharing_penalty.png`.

**Consequence for the study design.** `shared_encoder=True` remains the right control, and
the conflation risk at the top of this section still stands — but what SSL could be
credited with recovering on pixels is *learning speed*, not *final return*. An SSL arm that
matches baseline final score has recovered nothing; the claim has to be made on AUC or
steps-to-threshold.

Statistics: complete separation at 3v3 is Mann-Whitney p≈0.05 one-tailed / 0.10 two-tailed
— suggestive, not conclusive. The +43% effect size is what makes it credible. The doc's
own 5-seed minimum would settle it (~50 min).

`detach` with **no SSL attached** is a starved configuration, not a candidate control: the
encoder is shaped by the value loss alone. Both the −84% result and the "applies to SSL
arms only" scoping decision above are consistent — do not read 1.71 as evidence about
stop-gradient, only as the floor when nothing supplies a representation signal.

---

## Step 1: what was changed

### `RL/encoders.py` (new)
`MLPEncoder` / `CNNEncoder` exposing `output_dim`; `CNNEncoder` also exposes
`conv_out_shape` (SSL decoders need it). `ActorCritic` = encoder + policy head + value head,
flags `shared_encoder`, `detach_actor_encoder`, `feature_dim`.

Split is exact vs `common_networks.py`: unshared `ActorCritic` param counts match the
original two networks (MLP 9,220; CNN 678,471).

**Those numbers are config-specific — they are not universal constants.** CNN 678,471 is
`obs=(3,64,64), act=(3,3)`. For FlappyBirdImg `(4,64,64), act=(2,)` the correct figure is
**681,539** (shared: 341,155). Parity was re-verified empirically at both shapes. Anyone
re-running parity on a different env will see a "mismatch" and think the refactor broke;
it did not.

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
configs train; CSV written.

### Reproducibility — the "same seed → bit-identical" claim was WRONG

Measured directly:

| config | result |
|---|---|
| cuda, `deterministic_torch=False` (**the default**) | **DIVERGES**, max delta 1.4e-02 over 3 iters |
| cuda, `deterministic_torch=True` | bit-identical |
| cpu | bit-identical |

So runs are **not** reproducible on the default GPU path. By iteration 3 the same-seed
entropy envelope is ~0.02–0.03 — large enough that an earlier "regression" against a
stored reference looked like a real behavioral change and was not. Anything comparing two
runs for equality must either set `deterministic_torch=True` or use a tolerance above
this floor.

This is acceptable for the study (conclusions rest on seed-averaged behavior), but it
means a *specific* run cannot be reproduced exactly unless determinism is switched on.

Note: iteration-1 scores are identical across seeds because `policy_head.weight.div_(100)`
makes the initial policy uniform to within 0.004 (H = ln 3 exactly). Not a seeding bug.

### Entropy bug evidence
Isolated entropy term, from a collapsed policy: `+` sign drove H 0.8652 → 0.0149;
`−` sign restored 0.8652 → 1.0985 (= ln 3). The `+` was an exploration *penalty*.

Loose end: even fixed, `ent_coef` 0.0 vs 0.05 barely differs on CartPole (H 1.046 vs 1.048)
— policy gradient dominates. Mechanism is correct; may want retuning on harder envs.

---

## Remaining build order

2. ~~`SSLReplayBuffer` + `AuxTask` interface~~ **DONE**
3. ~~Vertical slice: L1, magnitude pruning, autoencoder, contrastive~~ **DONE**
4. ~~Metrics module~~ **DONE**
5. Remaining methods → Tier 1 sweep (**RUNNING**) → Tier 2 headline.

---

## Steps 2–4: what was built

### `RL/ssl_buffer.py` — `SSLReplayBuffer`
uint8 for images / float32 for vectors (auto, overridable). Stores
`(obs_t, obs_{t+k}, action_t)`; `k` is per-`sample()` so one buffer serves every family,
and `k=0` means "single frames" so reconstructive tasks are not needlessly restricted to
non-terminal steps. Pair validity excludes **both** episode boundaries and ring-buffer
wrap past the write head. Verified by encoding the timestep into pixel values and decoding
it back out of sampled pairs — gaps are exactly `k`, always.

### `RL/aux_tasks.py` — `AuxTask`
`NoAuxTask` (null object), `AutoencoderTask`, `ContrastiveTask` (CURL), DrQ `random_shift`.
The autoencoder decoder mirrors the encoder with `output_padding` **solved per layer**, so
it inverts any input shape rather than only 64×64.

**CURL temperature defaults to 1.0, i.e. off.** The bilinear `W` already learns its own
scale, so a second temperature just multiplies it — and because these logits are
unnormalised they grow with ‖feat‖², so a 0.1 temperature compounds that by another 10×.
Kept as a sweepable parameter; do not treat it as free.

Unclipped, the contrastive loss transiently blows up (observed peak 5.3 → final 0.98).
Clipped as `ssl_update` does: peak 4.3 → final 0.06. The clip is load-bearing, not hygiene.

### `RL/sparsity.py` — `SparsityMethod`
`NoSparsity`, `L1Regularization`, `MagnitudePruning` (Zhu–Gupta cubic, global threshold).
Masks re-applied after **every** optimizer step — verified that skipping this lets momentum
and weight decay resurrect pruned weights (sparsity → 0.0).

Two traps found the hard way:
- **`end_step` must be reachable within the run's update count.** A schedule ending at
  step 100 in a run that only performs 28 updates silently plateaus at `schedule(20)`.
  Confirmed exactly: measured 0.3904 vs predicted 0.3904.
- **Masks must follow the parameter's device.** `set_training_mode` moves the net
  cpu↔cuda every iteration; masks built on one device fail on the next call.

### `RL/metrics.py`
Entropy effective rank (default), `srank_δ`, dormant ratio (τ=0.025, normalised so it is
scale-invariant), measured sparsity, `HeldOutBatch`. Validated against known-rank matrices:
rank-1 → 1.00, rank-4 → 3.94, isotropic-32 → 31.9.

### `RL/PPO.py` wiring
New kwargs: `aux_task`, `sparsity`, `ssl_coef`, `ssl_updates_per_rollout`,
`ssl_batch_size`, `ssl_buffer_capacity`, `ssl_lr`, `metrics_every`, `metrics_batch`.
SSL heads join the existing optimizer as a param group (torch skips params whose `.grad`
is None, and every path zero-grads with `set_to_none=True`, so each path moves only what
it touched).

**`_shared_backward` now has a third independently-clipped path for the weight
regulariser** — same reasoning as the actor/critic split. An L1 penalty's gradient is
`coef·sign(w)` everywhere, so its norm is `coef·√n`; at `coef=1e-3` over 340k weights that
is **0.583, already above `max_grad_norm=0.5` on its own**. Folded into the actor path it
would consume the whole clip budget and starve the policy — the exact failure that method
exists to prevent. Measured, not assumed.

### `RL/env_utils.py` — `FrameStackWrapper` (new)
Grayscale + resize + k-frame stack → `(k, H, W)`. **Atari needs this**: `PixelObsWrapper`
returns one RGB frame, and a feedforward policy cannot recover the Breakout ball's
direction from a single frame. Without it every arm is handicapped equally and the
differences under study compress toward zero. Grayscale keeps channels at `k` not `3k`,
making the shape identical to FlappyBirdImg's `(4,64,64)` — same encoder, same decoders,
no special-casing.

### SSL set expanded to 6 methods (`RL/aux_tasks.py`)

Beyond autoencoder + contrastive, four more added, one decoder chain and one EMA
scaffold shared across the families exactly as the grouping below intends:

- **VQ-VAE** — autoencoder + a **product-VQ** codebook. A single 256-d vector quantized
  against one codebook would collapse the state to K symbols, so the vector is split into
  groups, each quantized (K^groups effective codes). Straight-through gradient to the
  encoder; codebook perplexity is logged to confirm the codebook stays used.
- **Masked VQ-VAE** — VQ-VAE + MAE-style patch masking: encode the masked frame,
  reconstruct the **original**. A per-patch MAE loss would need per-patch tokens the
  global-pooled CNN encoder does not produce, so it reconstructs the whole frame.
- **JEPA** — predict the EMA-target embedding of one view from a predictor on the online
  embedding of another, **in latent space, no negatives**. Reuses the ContrastiveTask EMA
  scaffold; collapse is held off by EMA-target + predictor asymmetry (BYOL mechanism), so
  effective rank is the arm to watch. I-JEPA's spatial mask-predict does not map onto a
  single-vector encoder — this is the tractable global-vector JEPA.
- **Ladder network** — the invasive outlier. Needs the encoder's **intermediate**
  activations, so it forwards through `encoder.cnn`/`fc` by hand (no hooks, no encoder
  edits). Clean pass → targets, noise-corrupted pass → gradient path, top-down decoder
  with lateral connections denoises every layer.

  **The naive ladder objective diverged** — the encoder inflated activation norms without
  bound (top-vector norm 1 → >1000) because it denoises toward its *own* activations, a
  moving target it can win by rescaling. The original ladder controls this with batch-norm
  at every layer; **standardising both sides per layer** (compare pattern, not scale) is
  the same fix and is stable. Caught by a "loss decreases when trained" test that instead
  showed the loss *growing* — worth having.

All four verified before the sweep: loss decreases when trained; VQ codebook stays used
(perplexity > 1); JEPA does not collapse (effective rank 85 of 256 after training); ladder
reaches conv activations and pushes gradient to the encoder; and **`ssl_lr` is a real
influence axis for each** (167–269× encoder movement over a 1000× lr range) — the check
that would have caught the inert-`ssl_coef` bug had it existed then.

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

## Step 5: the running ALE sweep

`ALE/Breakout-v5`, 4-frame grayscale stack at 64×64, 4M frames/run (166 iters @
n_steps=2000 × n_envs=12), 5 seeds, 17 conditions = **85 runs ≈ 59h**.

| family | levels |
|---|---|
| magnitude pruning | 90%, 95%, 99%, **99.9%** target sparsity |
| L1 | coef 1e-6, 1e-5, 1e-4, 1e-3 |
| autoencoder | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |
| contrastive (CURL) | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |
| VQ-VAE | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |
| Masked VQ-VAE | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |
| JEPA | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |
| ladder network | **ssl_lr** 1e-6, 1e-5, 1e-4, 1e-3 |

Full grid: 33 conditions × 5 seeds = 165 runs. The first 85 (baseline + sparsity + AE +
CURL) are done; the 80 added by the four new SSL families are running (~54h). Resumable —
completed cells are skipped, so this builds on the existing results rather than redoing
them.

### Why SSL is swept over `ssl_lr` and NOT a loss coefficient

**An auxiliary loss coefficient is inert when SSL takes its own optimizer step.** RMSprop
and Adam update by `lr·g/√v`, so scaling the loss by `c` scales `g` and `√v` alike and the
ratio cancels. Measured on the encoder's actual weight change:

| knob | 1000× range produces | verdict |
|---|---|---|
| `ssl_coef` | 1.1–3.6× movement, **inverted** (clipping caps the top) | inert |
| `ssl_lr` *(before fix)* | 1–4× | inert — it only reached the aux head |
| `ssl_lr` *(after fix)* | **235× (AE), 671× (CURL)**, monotone | works |

This was caught mid-sweep: the autoencoder AUC was flat across a 1000× coefficient range
(4.52/4.47/4.62/4.70) and `curl` at coef 0.01 — supposedly a near-no-op — ran far below
baseline. ~27h of the first sweep was measuring an axis that did not vary.

**The sparsity half was never affected, and the contrast is the proof.** L1's gradient is
*summed with* the actor and critic gradients before one shared step, so its coefficient
changes the gradient's composition rather than its scale — which is exactly why L1 shows a
clean monotone dose–response and a visible dose-dependent takeoff delay while SSL showed
nothing.

**The fix follows CURL.** CURL has no loss coefficient anywhere; it creates two optimizers
and steps them after a single contrastive backward:

```python
self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=encoder_lr)
self.cpc_optimizer     = torch.optim.Adam(self.CURL.parameters(),           lr=encoder_lr)
```

The original implementation here had only the second of those — `ssl_lr` was attached to
the aux head alone, so the shared trunk moved at the *policy* `lr` regardless. `PPO` now
builds `opt_ssl` over **encoder + aux heads** at `ssl_lr` (`_build_ssl_optimizer`). The
encoder consequently sits in two optimizers with independent state; that is deliberate in
CURL too, so RL and representation learning can step the same weights at different rates.

`ssl_coef` is retained (it still sets where the SSL gradient meets `max_grad_norm`) but it
is **not** the influence variable. Verified after the change: baseline still bit-identical
to pristine.

Runs from the first sweep whose axis was inert are in `runs_ale_archive/`, not deleted —
they remain valid replicates of a single condition (autoencoder at 8 updates/rollout).

**Sparsity levels apply to magnitude pruning only.** L1 has no sparsity target — it shrinks
weights without zeroing them — so it is swept over its coefficient and its *achieved*
sparsity reported (≈0 exact zeros). Reporting a requested level L1 never reaches would be
a fiction; see `fig5_measured_sparsity.png`.

**Levels were chosen by measuring effective gradient norm after clipping**, so no two arms
are secretly identical. All levels produce distinct updates; the **top level of each sweep
saturates `max_grad_norm=0.5`** and means "auxiliary term dominates" — going higher would
produce an identical arm.

| coef | AE raw grad | CURL raw grad | L1 raw grad |
|---|---|---|---|
| low | 0.0016 | 0.0007 | 0.0006 |
| … | 0.0159 / 0.1592 | 0.0076 / 0.0756 | 0.0058 / 0.0583 |
| top | 1.5921 → **clipped** | 0.7785 → **clipped** | 0.5831 → **clipped** |

Pruning schedule `start=500, end=4000` of ~9.4k updates — pruning completes ~43% through
training, leaving the majority of the run to recover.

Throughput measured end-to-end: **1734 fps baseline / ~1650 with SSL** (SSL overhead only
~3%; the network forward during collection dominates, not the emulator).

---

## Environment notes

- Python 3.11, **torch 2.6.0+cu124** (the earlier "2.11.0+cu130" was wrong), CUDA
  available, 16 CPUs, RTX 4070 Laptop 8.6 GB, 34 GB RAM.
- `gymnasium 0.29.1`, `ale-py 0.8.1`, `minatar 1.0.15` now **installed**.
  `numpy`, `opencv-python`, `matplotlib`, `scipy` present.
- **MinAtar does not work with `CNNEncoder` as-is**: it is 10×10, and the k8/s4 first conv
  collapses that to 1×1, after which the k4 conv fails. It only appears to work through
  `PixelObsWrapper`, which silently upscales 10×10 → 64×64 — wasted compute, not a real
  Tier 1. Using it needs the mini-CNN branch to trigger on spatial dims.
- **Any script spawning env workers needs `if __name__ == '__main__':`** (Windows spawn).
- Use `python -u` for scripts whose output is redirected, or nothing appears until exit.

Verification scripts live in the session scratchpad (not the repo): `entropy_gradient_check.py`,
`refactor_check.py`, `diagnose.py`, `grad_balance.py`, `optimizer_ablation.py`, `final_verify.py`.
They are regenerable from this document if lost.

Session 2 scratchpad scripts: `sharing_penalty_flappy.py` + `run_batch.py` (the pixel
sharing grid), `test_step234.py` (45 unit tests), `test_integration.py` (end-to-end),
`compare_pristine.py` (**the real regression test** — pristine HEAD vs edited PPO under
cudnn-deterministic, bit-identical), `test_determinism.py`, `check_coef_range.py`
(clip-saturation measurement), `bench_envs.py` / `bench_ale_e2e.py`, `sweep_ale.py` +
`run_sweep.py`, `plots.py`, `analyze.py`.
