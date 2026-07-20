import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import scipy, math
import matplotlib.pyplot as plt

from RL.buffers import RolloutBuffer
from RL.vec_env_handler import ParallelEnvManager
from RL.encoders import ActorCritic
from RL.experiment_utils import set_global_seed, RunLogger
from RL.ssl_buffer import SSLReplayBuffer
from RL.aux_tasks import NoAuxTask, make_aux_task
from RL.sparsity import NoSparsity, make_sparsity
from RL.metrics import HeldOutBatch, representation_report

from RL.common_networks import (
    base_MLP_model,
    base_CNN_model,
)
from RL.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)


class PPO():

    def __init__(
            self,
            env, # callable that creates an instance of the env, None for no env
            observation_space, # number of values in input
            action_space, # number of values in output
            lr = 1e-4, # learning rate of actor (and shared encoder)
            value_lr = 3e-4, # learning rate of critic (larger than actor)
            n_steps = 4000, # number of steps to train per batch of games
            batch_size = 250, # minibatch size
            epochs = 7, # number of epochs
            clip_range = 0.2, # clip range for PPO
            discount = .99, # discount rate
            gae_lambda = 0.95, # td lambda for GAE
            normalize_advantage = True, # normalize advantage (in this case returns)
            ent_coef = 5e-3, # entropy coefficient
            vf_coef = 0.5, # value loss coefficient (shared-encoder path only)
            max_grad_norm = 0.5, # max gradient norm when clipping
            verbose = True, # use print statements
            models = None, # default none, can specify model
            n_envs = 1, # vectorized env, how many environments to run in parallel, around #cpus
            device = "auto", # gpu or cpu
            shared_encoder = True, # single trunk feeding policy + value + SSL heads
            optimizer = "rmsprop", # "rmsprop" matches the original policy optimizer
            detach_actor_encoder = False, # stop policy gradients at the shared encoder
            feature_dim = None, # encoder width; None -> 64 (MLP) / 256 (CNN)
            seed = None, # seeds process RNGs and env workers; None = unseeded
            logger = None, # RunLogger instance, or None for no CSV logging
            aux_task = None, # AuxTask instance or name; None -> NoAuxTask (baseline)
            sparsity = None, # SparsityMethod instance or name; None -> NoSparsity (baseline)
            ssl_coef = 1.0, # weight on the auxiliary loss (independent variable, not a nuisance knob)
            ssl_updates_per_rollout = 8, # SSL grad steps per rollout; own cadence, not PPO's
            ssl_batch_size = 256,
            ssl_buffer_capacity = 4096, # timesteps PER ENV -> capacity * n_envs frames
            ssl_lr = None, # SSL param-group lr; None -> lr
            metrics_every = 0, # compute representation metrics every N iterations; 0 = off
            metrics_batch = 4096, # held-out batch size for effective rank / dormant ratio
        ):

        if isinstance(observation_space, int):
            observation_space = (observation_space,)
        if isinstance(action_space, int):
            action_space = (action_space,)

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.lr = lr
        self.value_lr = value_lr
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.n_envs = n_envs
        self.shared_encoder = shared_encoder
        self.optimizer_name = optimizer
        self.seed = seed
        self.logger = logger
        self.ssl_coef = ssl_coef
        self.ssl_updates_per_rollout = ssl_updates_per_rollout
        self.ssl_batch_size = ssl_batch_size
        self.ssl_buffer_capacity = ssl_buffer_capacity
        self.ssl_lr = ssl_lr if ssl_lr is not None else lr
        self.metrics_every = metrics_every

        if seed is not None:
            set_global_seed(seed)

        self.training_history = []
        self.num_timesteps = 0
        self.n_updates = 0

        if models:
            # legacy path: caller supplies (policy_net, value_net) as two separate modules.
            # No shared trunk exists, so SSL tasks cannot attach here.
            self.model, self.value_net = models
            self.ac = None
            self.shared_encoder = False
        else:
            self.ac = ActorCritic(
                observation_space=observation_space,
                action_space=action_space,
                shared_encoder=shared_encoder,
                detach_actor_encoder=detach_actor_encoder,
                feature_dim=feature_dim,
            )
            # aliases kept so existing save/load and demo code keeps working
            self.model = self.ac
            self.value_net = self.ac

        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps,
                                            observation_space=observation_space,
                                            action_dim=self.action_dim,
                                            device=device,
                                            gae_lambda=gae_lambda,
                                            discount=discount,
                                            n_envs=n_envs,
                                            )
        if self.action_dim == 1:
            self.distribution = CategoricalDistribution
        else:
            self.distribution = MultiCategoricalDistribution

        # --- SSL + sparsity ---------------------------------------------------------
        # Both default to null objects, so a baseline run executes the same code path as
        # an SSL/sparsity run with the method removed -- no `if ssl:` branching anywhere
        # that could quietly make the control condition a different algorithm.
        # accept either a constructed instance or a name string
        self.aux_task = aux_task if hasattr(aux_task, "loss") else make_aux_task(aux_task)
        self.sparsity = sparsity if hasattr(sparsity, "regularizer") else make_sparsity(sparsity)

        self.has_ssl = not isinstance(self.aux_task, NoAuxTask)
        self.has_sparsity = not isinstance(self.sparsity, NoSparsity)

        if self.has_ssl:
            if self.ac is None:
                raise RuntimeError(
                    "SSL tasks need a shared encoder; PPO was constructed with models=..."
                )
            if not self.shared_encoder:
                raise RuntimeError(
                    "SSL tasks need shared_encoder=True (nothing to attach to otherwise)"
                )
            self.aux_task.build(self.ac.encoder, observation_space, action_space, self.device)

        # The buffer is also the observation source for the held-out metrics batch, so a
        # baseline run with metrics enabled still needs one -- just big enough to fill it.
        if self.has_ssl:
            capacity = ssl_buffer_capacity
        elif metrics_every:
            capacity = max(64, -(-metrics_batch // max(n_envs, 1)))
        else:
            capacity = 0

        self.ssl_buffer = SSLReplayBuffer(
            capacity=capacity,
            observation_space=observation_space,
            action_dim=self.action_dim,
            n_envs=n_envs,
        ) if capacity else None

        self.modules_dict = self._build_modules_dict()
        self.sparsity.on_init(self.modules_dict)

        self.held_out = HeldOutBatch(n=metrics_batch, device=self.device) \
            if metrics_every else None

        self._build_optimizers()

        if env != None:
            self.env_manager = ParallelEnvManager(self.env, self.n_envs, seed=seed)
            self.last_obs = self.env_manager.reset()

    def _build_modules_dict(self):
        '''
        Named view of the network for SparsityMethod. Names matter: _prunable() excludes
        'policy_head'/'value_head' unless include_heads=True.

        The legacy models=(policy, value) path exposes whole Sequentials, whose final
        Linear IS the output layer -- so include_heads has no effect there and pruning
        would reach the output. SSL already refuses that path; sparsity on it is
        untested and not part of the study.
        '''
        if self.ac is None:
            return {"policy": self.model, "value": self.value_net}
        return {
            "encoder": self.ac.encoder,
            "value_encoder": self.ac.value_encoder,
            "policy_head": self.ac.policy_head,
            "value_head": self.ac.value_head,
        }

    def _ssl_params(self):
        '''Params the SSL loss is allowed to move: shared trunk + the task's own heads.'''
        return list(self.ac.encoder.parameters()) + list(self.aux_task.parameters())

    def _build_optimizers(self):
        '''
        Shared encoder needs a single optimizer: policy and value losses both touch the
        trunk, so two separate .backward() calls would hit a freed graph. Parameter
        groups preserve the original two-learning-rate setup (lr for the actor path,
        value_lr for the critic head).

        The legacy unshared path keeps the original RMSprop/Adam pair verbatim so it
        remains a faithful regression baseline.
        '''
        if self.ac is None or not self.shared_encoder:
            if self.ac is None:
                policy_params = self.model.parameters()
                value_params = self.value_net.parameters()
            else:
                policy_params = self.ac.actor_parameters()
                value_params = self.ac.critic_parameters()
            self.opt = torch.optim.RMSprop(policy_params, lr=self.lr, weight_decay=1e-5)
            self.opt_value_net = torch.optim.Adam(value_params, lr=self.value_lr, weight_decay=1e-5)
            self.single_optimizer = False
        else:
            # RMSprop by default: measured on CartPole, switching the policy optimizer to
            # Adam cost ~34% of learning progress, so the shared path keeps the algorithm
            # the original policy optimizer used. Parameter groups preserve the two
            # learning rates.
            groups = [
                {"params": list(self.ac.encoder.parameters()), "lr": self.lr},
                {"params": list(self.ac.policy_head.parameters()), "lr": self.lr},
                {"params": list(self.ac.value_head.parameters()), "lr": self.value_lr},
            ]
            cls = torch.optim.Adam if self.optimizer_name == "adam" else torch.optim.RMSprop
            self.opt = cls(groups, weight_decay=1e-5)
            self.opt_value_net = None
            self.single_optimizer = True

        self._build_ssl_optimizer()

    def _build_ssl_optimizer(self):
        '''
        SSL gets its OWN optimizer over the shared trunk plus its own heads, at `ssl_lr`.
        This follows CURL, which creates exactly two:

            encoder_optimizer = Adam(critic.encoder.parameters(), lr=encoder_lr)
            cpc_optimizer     = Adam(CURL.parameters(),           lr=encoder_lr)

        and steps both after a single contrastive backward, with NO loss coefficient
        anywhere. The encoder therefore sits in two optimizers with independent state --
        deliberate in CURL, not an oversight: RL and representation learning step the
        same weights at different rates.

        Why this matters here. An auxiliary LOSS COEFFICIENT cannot control SSL influence
        when SSL takes its own optimizer step: RMSprop and Adam update by lr*g/sqrt(v),
        so scaling the loss by c scales g and sqrt(v) alike and the ratio cancels.
        Measured: a 1000x `ssl_coef` range moved the encoder only 1.1-3.6x, and inverted
        (clipping caps the high end). `ssl_lr` multiplies AFTER that normalisation, so it
        does scale the step -- it is the real knob, and it is the one CURL exposes.

        `ssl_coef` is retained because it still sets where the gradient meets
        max_grad_norm, but it is NOT the influence variable. Sweep `ssl_lr`.
        '''
        self.opt_ssl = None
        if not self.has_ssl:
            return
        cls = torch.optim.Adam if self.optimizer_name == "adam" else torch.optim.RMSprop
        self.opt_ssl = cls(self._ssl_params(), lr=self.ssl_lr, weight_decay=1e-5)

    def parameters(self):
        if self.ac is not None:
            return self.ac.parameters()
        return list(self.model.parameters()) + list(self.value_net.parameters())

    def set_training_mode(self, training_mode):
        target = self.device if training_mode else "cpu"
        if self.ac is not None:
            self.ac.to(target)
        else:
            self.model.to(target)
            self.value_net.to(target)

    def train(self):
        self.set_training_mode(True)
        clip_range = self.clip_range

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
                 "clip_fraction": [], "sparsity_reg": []}

        for epoch in range(self.epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.log_probs)

                # clipped loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
                policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2))

                # entropy bonus for exploration. NOTE the minus: the loss is MINIMIZED,
                # so a positive coefficient here would drive entropy toward zero and
                # collapse the policy to deterministic. This was previously a plus.
                entropy_loss = torch.mean(entropy)

                value_loss = F.mse_loss(values, rollout_data.returns)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.log_probs
                    stats["approx_kl"].append(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item())
                    stats["clip_fraction"].append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_loss.item())

                # Weight-space penalty (L1/L0). Data-independent, so it rides along with
                # the actor path rather than getting its own backward.
                reg = self.sparsity.regularizer(self.modules_dict)
                if reg is not None:
                    stats["sparsity_reg"].append(reg.item())

                if self.single_optimizer:
                    self._shared_backward(policy_loss, entropy_loss, value_loss, reg)
                else:
                    loss = policy_loss - self.ent_coef * entropy_loss
                    if reg is not None:
                        loss = loss + reg

                    # policy and value sit on independent trunks here, so their graphs
                    # are disjoint and no retain_graph is needed
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._policy_params(), self.max_grad_norm)
                    self.opt.step()

                    self.opt_value_net.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self._value_params(), self.max_grad_norm)
                    self.opt_value_net.step()

                self.n_updates += 1
                # after EVERY optimizer step: re-apply masks and advance the schedule,
                # or momentum/weight-decay will resurrect pruned weights
                self.sparsity.after_step(self.modules_dict, self.n_updates)

        self.set_training_mode(False)
        # sparsity_reg is empty on baseline runs; None keeps the CSV column present
        return {k: (float(np.mean(v)) if v else None) for k, v in stats.items()}

    def _shared_backward(self, policy_loss, entropy_loss, value_loss, reg=None):
        '''
        Shared trunk, but the actor and critic paths are clipped INDEPENDENTLY.

        Naively summing the losses and applying one global clip_grad_norm_ is wrong here:
        the value loss is on a far larger scale than the policy loss (measured at ~500x
        the trunk gradient norm on CartPole), so a single global clip scales the policy
        signal down by whatever factor the value gradient demands, and the actor barely
        learns. The legacy two-optimizer path did not have this problem because each
        network was clipped on its own.

        So: backward each path separately, clip each to max_grad_norm, accumulate, step
        once. Costs one extra backward per minibatch, which is negligible next to env
        stepping. When SSL is added it becomes a third independently-clipped path.
        '''
        params = [p for p in self.ac.parameters() if p.requires_grad]

        actor_loss = policy_loss - self.ent_coef * entropy_loss
        critic_loss = self.vf_coef * value_loss

        def backward_and_clip(loss, retain):
            '''One independently-clipped path: returns its clipped gradients.'''
            self.opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=retain)
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            return [None if p.grad is None else p.grad.detach().clone() for p in params]

        # The weight regulariser gets its OWN clip for the same reason the critic does.
        # An L1 penalty's gradient is coef*sign(w) everywhere, so its norm is
        # coef*sqrt(n_params) -- at coef=1e-3 over 340k weights that is ~0.58, already
        # above max_grad_norm on its own. Folded into the actor path it would consume
        # the entire clip budget and scale the policy signal to nearly nothing, which is
        # exactly the failure this method exists to avoid.
        paths = [backward_and_clip(actor_loss, retain=True)]
        if reg is not None:
            paths.append(backward_and_clip(reg, retain=True))
        paths.append(backward_and_clip(critic_loss, retain=False))

        self.opt.zero_grad(set_to_none=True)
        for grads in paths:
            for p, g in zip(params, grads):
                if g is None:
                    continue
                p.grad = g if p.grad is None else p.grad + g

        self.opt.step()

    def ssl_update(self):
        '''
        SSL runs on its own cadence off the replay buffer -- NOT inside PPO's epoch loop
        over on-policy rollouts. Decoupling them is what lets the SSL task see old data
        many times while PPO sees each rollout a fixed number of times, which is most of
        the sample-efficiency argument for auxiliary tasks in the first place.

        This is the third independently-clipped gradient path described in
        _shared_backward: its own zero_grad / backward / clip / step, so the SSL gradient
        cannot be rescaled by whatever the value gradient happens to be doing.

        Returns mean SSL loss, or None when nothing ran.
        '''
        if not self.has_ssl or self.ssl_updates_per_rollout <= 0:
            return None

        k = self.aux_task.k_step
        if not self.ssl_buffer.can_sample(self.ssl_batch_size, k):
            return None  # buffer still warming up

        self.set_training_mode(True)
        self.aux_task.to(self.device).train(True)

        losses = []
        for _ in range(self.ssl_updates_per_rollout):
            batch = self.ssl_buffer.sample(self.ssl_batch_size, k=k, device=self.device)
            loss = self.aux_task.loss(batch, self.ac.encoder)
            if loss is None:
                break

            # steps opt_ssl, NOT opt: the encoder must move at ssl_lr here, at lr on the
            # RL path. Sharing one optimizer is what made ssl_lr reach only the aux head.
            self.opt_ssl.zero_grad(set_to_none=True)
            (self.ssl_coef * loss).backward()
            nn.utils.clip_grad_norm_(self._ssl_params(), self.max_grad_norm)
            self.opt_ssl.step()

            self.aux_task.after_step()      # EMA / momentum bookkeeping
            self.n_updates += 1
            self.sparsity.after_step(self.modules_dict, self.n_updates)

            losses.append(loss.item())

        self.set_training_mode(False)
        return float(np.mean(losses)) if losses else None

    def compute_metrics(self):
        '''Representation health on the fixed held-out batch. None until it is filled.'''
        if self.held_out is None or self.ssl_buffer is None:
            return None
        if not self.held_out.fill_from(self.ssl_buffer):
            return None

        self.set_training_mode(True)
        report = representation_report(
            self.ac.encoder, self.held_out.obs.to(self.device), self.modules_dict
        )
        self.set_training_mode(False)
        return report

    def _policy_params(self):
        if self.ac is not None:
            return self.ac.actor_parameters()
        return self.model.parameters()

    def _value_params(self):
        if self.ac is not None:
            return self.ac.critic_parameters()
        return self.value_net.parameters()

    # forward actor and critic
    def forward(self, obs):
        if self.ac is not None:
            logits, values, _ = self.ac(obs)
            return logits, values
        actions = self.model(obs)
        values = self.value_net(obs)
        return actions, values

    def encode(self, obs):
        '''Shared representation, for SSL auxiliary tasks and representation metrics.'''
        if self.ac is None:
            raise RuntimeError("no shared encoder: PPO was constructed with models=...")
        return self.ac.encode(obs)

    def get_action(self, obs):
        action_logits, value = self.forward(torch.from_numpy(obs))
        distribution = self.distribution(self.action_space, action_logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def get_values(self, obs):
        if self.ac is not None:
            _, values, _ = self.ac(torch.from_numpy(obs))
            return values
        return self.value_net(torch.from_numpy(obs))

    def evaluate_actions(self, observations, actions):
        action_logits, values = self.forward(observations)
        distribution = self.distribution(self.action_space, action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    # main training loop
    def learn(self, total_steps, progress_bar=True):
        num_steps = 0
        while num_steps < total_steps:
            score = self.collect_rollouts(progress_bar=progress_bar)
            num_steps += self.rollout_buffer.size()
            self.num_timesteps += self.rollout_buffer.size() * self.n_envs
            stats = self.train()
            ssl_loss = self.ssl_update()

            if self.verbose: print(round(score,3))
            else: print()
            self.training_history.append(score)
            iteration = len(self.training_history)

            if self.logger is not None:
                # RunLogger fixes its columns on the FIRST log() call, so every metric
                # must appear from iteration 1 even when its value is not available yet
                # -- otherwise it is silently dropped for the whole run.
                metrics = None
                if self.metrics_every and iteration % self.metrics_every == 0:
                    metrics = self.compute_metrics()
                metrics = metrics or {}

                self.logger.log(
                    iteration=iteration,
                    env_steps=self.num_timesteps,
                    n_updates=self.n_updates,
                    score=float(score),
                    ssl_loss=ssl_loss,
                    measured_sparsity=self.sparsity.measured_sparsity(self.modules_dict)
                        if self.has_sparsity else None,
                    eff_rank=metrics.get("eff_rank"),
                    srank=metrics.get("srank"),
                    dormant_ratio=metrics.get("dormant_ratio"),
                    **stats,
                )

    def collect_rollouts(self, progress_bar):
        self.rollout_buffer.reset()

        progress = 0
        if progress_bar:
            print('#',end='')

        total_rewards, total_dones = 0, 0

        while not self.rollout_buffer.full:
            self.last_obs = np.array(self.last_obs, dtype=np.float32)

            with torch.no_grad():
                actions, log_probs, values = self.get_action(self.last_obs)
            new_obs, rewards, dones = self.env_manager.step(np.array(actions))

            self.rollout_buffer.add(
                self.last_obs,
                actions,
                rewards,
                dones,
                values,
                log_probs,
            )
            # SSL sees the same stream but keeps it across rollouts (uint8, off-policy).
            # dones here marks termination of the transition OUT of last_obs, which is
            # exactly the convention SSLReplayBuffer assumes for pair validity.
            if self.ssl_buffer is not None:
                self.ssl_buffer.add(self.last_obs, actions, dones)
            total_rewards += sum(rewards)
            total_dones += abs(sum(dones))

            self.last_obs = new_obs

            if progress_bar:
                new_progress = self.rollout_buffer.progress()//.1
                if progress < new_progress:
                    print('#',end='')
                    progress = new_progress

        self.last_obs = np.array(self.last_obs, dtype=np.float32)
        with torch.no_grad():
            values = self.get_values(self.last_obs)

        self.rollout_buffer.compute_return_and_advantage(values, dones)
        return total_rewards/max(total_dones,1)

    def test(self, n_steps=1_000, **kwargs):
        cumulative_reward = 0
        env = self.env(**kwargs)
        obs, info = env.reset()
        for step in range(n_steps):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                action, action_log_prob, _ = self.get_action(obs)
            if self.action_dim == 1: action = action.item()
            else: action = action.tolist()[0]
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cumulative_reward += reward

            if done: break
            obs = new_obs

        env.close()

        return cumulative_reward

    def plot_training_history(self, step=20):
        training_history_smoothed = []
        for i in range(0, len(self.training_history), step):
            training_history_smoothed.append(np.average(self.training_history[i:i+20]))

        plt.plot(list(range(len(self.training_history))), self.training_history, alpha=0.3)
        plt.plot([i*20 for i in range(len(training_history_smoothed))], training_history_smoothed)
        plt.show()
