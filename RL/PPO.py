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

        self._build_optimizers()

        if env != None:
            self.env_manager = ParallelEnvManager(self.env, self.n_envs, seed=seed)
            self.last_obs = self.env_manager.reset()

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

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

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

                if self.single_optimizer:
                    self._shared_backward(policy_loss, entropy_loss, value_loss)
                else:
                    loss = policy_loss - self.ent_coef * entropy_loss

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

        self.set_training_mode(False)
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _shared_backward(self, policy_loss, entropy_loss, value_loss):
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

        self.opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        actor_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        for p, g in zip(params, actor_grads):
            if g is None:
                continue
            p.grad = g if p.grad is None else p.grad + g

        self.opt.step()

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

            if self.verbose: print(round(score,3))
            else: print()
            self.training_history.append(score)

            if self.logger is not None:
                self.logger.log(
                    iteration=len(self.training_history),
                    env_steps=self.num_timesteps,
                    n_updates=self.n_updates,
                    score=float(score),
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
