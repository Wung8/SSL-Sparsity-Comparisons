import numpy as np
import torch


class SSLBatch:
    '''Container for one SSL minibatch. obs/next_obs are float32 in [0,1] on device.'''

    def __init__(self, obs, next_obs, actions):
        self.obs = obs
        self.next_obs = next_obs
        self.actions = actions


class SSLReplayBuffer:
    '''
    Off-policy store feeding the self-supervised auxiliary tasks.

    Deliberately NOT a copy of RolloutBuffer:

    * **uint8 storage for pixels.** 100k frames of 4x64x64 is 1.6 GB as uint8 and 6.6 GB
      as float32. RolloutBuffer's all-float32 pattern does not scale to replay sizes.
      Non-image observations are kept float32 -- quantising an unbounded state vector to
      256 levels would destroy it -- so compression is opt-in via `compress`.

    * **Transitions, not frames.** Joint-embedding / JEPA tasks need the pair
      (obs_t, obs_{t+k}); action-conditioned variants also need action_t. Reconstructive
      tasks ignore the pair and read obs_t only. k is configurable per sample() call so
      one buffer serves every task family.

    * **Episode- and wrap-safe pair sampling.** Two ways a naive obs[t+k] goes wrong:
      the env workers auto-reset, so the observation following a done belongs to a fresh
      episode; and once the ring buffer wraps, indices past the write head are unrelated
      older data. Both are excluded in _valid_starts().

    Done semantics, matching PPO.collect_rollouts: dones[t] marks that the transition
    out of obs[t] terminated the episode. So obs[t] and obs[t+1] are same-episode iff
    dones[t] == 0, and a k-step pair starting at t is valid iff dones[t..t+k-1] are all 0.
    '''

    def __init__(
        self,
        capacity,              # timesteps stored per env (total frames = capacity * n_envs)
        observation_space,
        action_dim,
        n_envs,
        device="cpu",
        compress=None,         # None -> auto: uint8 for image observations only
    ):
        if isinstance(observation_space, int):
            observation_space = (observation_space,)
        self.capacity = capacity
        self.observation_space = tuple(observation_space)
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = device

        if compress is None:
            # images only: (C,H,W) already scaled to [0,1] by PixelObsWrapper
            compress = len(self.observation_space) == 3
        self.compress = compress

        obs_dtype = np.uint8 if compress else np.float32
        self.observations = np.zeros((capacity, n_envs, *self.observation_space), dtype=obs_dtype)
        self.actions = np.zeros((capacity, n_envs, action_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def size(self):
        return self.capacity if self.full else self.pos

    def n_frames(self):
        return self.size() * self.n_envs

    def nbytes(self):
        return self.observations.nbytes + self.actions.nbytes + self.dones.nbytes

    def _encode(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if self.compress:
            # PixelObsWrapper emits [0,1]; round-trip error is 1/255 and irrelevant to SSL
            return np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        return obs

    def _decode(self, arr):
        if self.compress:
            return arr.astype(np.float32) / 255.0
        return arr

    def add(self, obs, action, done):
        '''One vectorised timestep across all envs. Overwrites oldest data once full.'''
        self.observations[self.pos] = self._encode(obs)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(self.n_envs, self.action_dim)
        self.dones[self.pos] = np.asarray(done, dtype=np.float32)

        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    def _to_physical(self, logical):
        '''
        Logical index 0 is the OLDEST retained timestep. Once wrapped, that is self.pos;
        before wrapping the buffer is not circular yet and logical == physical.
        '''
        if not self.full:
            return logical
        return (self.pos + logical) % self.capacity

    def _valid_starts(self, k):
        '''
        Boolean mask over (logical_start, env) for k-step pairs that stay inside one
        episode and inside the retained window.
        '''
        n = self.size()
        n_starts = n - k
        if n_starts <= 0:
            return None, 0

        starts = np.arange(n_starts)
        valid = np.ones((n_starts, self.n_envs), dtype=bool)
        # a done anywhere in [t, t+k-1] means obs[t+k] is from a later episode
        for j in range(k):
            phys = self._to_physical(starts + j)
            valid &= self.dones[phys] == 0
        return valid, n_starts

    def can_sample(self, batch_size, k=1):
        valid, n_starts = self._valid_starts(k)
        return valid is not None and valid.sum() >= batch_size

    def sample(self, batch_size, k=1, device=None):
        '''
        Returns an SSLBatch of float32 observations on `device`.

        k=0 is allowed and means "no pair needed" -- next_obs is then a copy of obs and
        every stored timestep is a valid start. Reconstructive tasks should use k=0 so
        they are not needlessly restricted to non-terminal steps.
        '''
        device = device or self.device

        if k == 0:
            n = self.size()
            if n == 0:
                raise ValueError("SSLReplayBuffer is empty")
            li = np.random.randint(0, n, size=batch_size)
            ei = np.random.randint(0, self.n_envs, size=batch_size)
            phys = self._to_physical(li)
            obs = self._decode(self.observations[phys, ei])
            act = self.actions[phys, ei]
            t = lambda a: torch.as_tensor(a, device=device)
            return SSLBatch(t(obs), t(obs.copy()), t(act))

        valid, n_starts = self._valid_starts(k)
        if valid is None or valid.sum() == 0:
            raise ValueError(f"no valid {k}-step transitions in SSLReplayBuffer")

        flat = np.flatnonzero(valid.reshape(-1))
        pick = np.random.randint(0, len(flat), size=batch_size)
        li, ei = np.unravel_index(flat[pick], valid.shape)

        phys_t = self._to_physical(li)
        phys_tk = self._to_physical(li + k)

        obs = self._decode(self.observations[phys_t, ei])
        next_obs = self._decode(self.observations[phys_tk, ei])
        act = self.actions[phys_t, ei]

        t = lambda a: torch.as_tensor(a, device=device)
        return SSLBatch(t(obs), t(next_obs), t(act))
