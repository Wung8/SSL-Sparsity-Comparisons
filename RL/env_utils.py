import cv2
import numpy as np


def make(env_name, **kwargs):
    if env_name == "Cartpole":
        from RL.environments.cartpole import CartPoleEnvironment as env
    if env_name == "CartpoleNoVel":
        from RL.environments.cartpole_novelocity import CartPoleEnvironment as env
    if env_name == "Diep":
        from RL.environments.diep import DiepioEnvironment as env
    if env_name == "FlappyBird":
        from RL.environments.flappy_bird import FlappyBirdEnvironment as env
    if env_name == "FlappyBirdImg":
        from RL.environments.flappy_bird_img import FlappyBirdEnvironment as env
    if env_name == "Platformer":
        from RL.environments.platformer import PlatformerEnvironment as env
    if env_name == "SimpleMemory":
        from RL.environments.simple_memory import SimpleMemoryEnvironment as env
    if env_name == "SlimeVolleyballTwoPlayer":
        from RL.environments.slime_volleyball import SlimeEnvironment as env
    if env_name == "SlimeVolleyball":
        from RL.environments.slime_volleyball_single_player import SlimeEnvironment as env
    if env_name == "Soccer":
        from RL.environments.soccer import SoccerEnvironment as env
    if env_name == "TMaze":
        from RL.environments.T_maze import TMazeEnvironment as env
    

    return env(**kwargs)
    

    

class SkipFrameWrapper:
    def __init__(self, env, frame_skip=4, **kwargs):
        self.env = env
        self.frame_skip = frame_skip
        
    def step(self, action):
        total_reward, terminated, truncated, info = 0, False, False, {}
        for t in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break
            
        return observation, total_reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
    

class FrameStackWrapper:
    '''
    Grayscale + resize + k-frame stack, emitting (k, H, W) float32 in [0,1].

    PixelObsWrapper alone is not enough for Atari: it returns a single RGB frame, from
    which velocity is unrecoverable. A feedforward policy cannot tell which way the
    Breakout ball is travelling from one frame, so every method would be handicapped
    equally and the differences under study would be compressed toward zero.

    Grayscale rather than stacked RGB keeps the channel count at k instead of 3k, which
    also makes the observation shape identical to FlappyBirdImg's (4, 64, 64) -- so the
    same encoder and the same SSL decoders apply without special-casing.
    '''

    def __init__(self, env, k=4, dsize=(64, 64), **kwargs):
        self.env = env
        self.k = k
        self.dsize = dsize
        self.stack = None

    def _frame(self, obs):
        obs = np.asarray(obs)
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, dsize=self.dsize, interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32) / 255.0

    def _stacked(self):
        return np.stack(self.stack, axis=0)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        f = self._frame(obs)
        # start-of-episode: repeat the first frame so the stack is never zero-padded,
        # which would otherwise look like a hard scene cut to the encoder
        self.stack = [f.copy() for _ in range(self.k)]
        info = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.stack.pop(0)
        self.stack.append(self._frame(obs))
        return self._stacked(), reward, terminated, truncated, info

    def close(self):
        self.env.close()


class PixelObsWrapper:
    def __init__(self, env, dsize=(64,64), **kwargs):
        self.env = env
        self.dsize = dsize
        
    def step(self, action):
        result = self.env.step(action)
        result = (self.edit_obs(result[0]), *result[1:])
        return result

    def reset(self):
        result = self.env.reset()
        result = (self.edit_obs(result[0]), *result[1:])
        return result

    def close(self):
        self.env.close()

    '''
    transpose output to move the channel dimension from index 2 to index 0. images are also
    downscaled to 64x64 and scaled to [0,1] to speed up training.
    '''
    def edit_obs(self, obs):
        obs = np.array(obs, dtype=np.float32)
        obs = cv2.resize(obs, dsize=self.dsize, interpolation=cv2.INTER_AREA)
        return np.transpose(obs, (2, 0, 1)) / 255
