import numpy as np
import torch
from torch import nn

'''
Encoders own everything up to (but not including) the final output layer, and expose
`output_dim` so that policy/value heads and SSL auxiliary heads can all be attached to
the same trunk.

The split is deliberately exact w.r.t. RL/common_networks.py:

  base_MLP_model  = Linear(obs,64) Mish Linear(64,64) Mish | Linear(64,out)
                    \___________ MLPEncoder ____________/   \___ head ___/

  base_CNN_model  = conv stack Linear(n_flat,256) Mish     | Linear(256,out)
                    \___________ CNNEncoder ___________/     \____ head ___/

so an unshared ActorCritic is architecturally identical to the original two networks.
'''


class MLPEncoder(nn.Module):
    def __init__(self, input_space, feature_dim=64):
        super().__init__()
        self.output_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_space[0], feature_dim),
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim),
            nn.Mish(),
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_space, feature_dim=256):
        super().__init__()
        n_input_channels = input_space[0]

        # mirrors base_CNN_model's choice of a smaller stack for tiny inputs
        if all(dim <= 7 for dim in input_space):
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.Mish(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.Mish(),
                nn.Flatten(start_dim=-3),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.Mish(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.Mish(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.Mish(),
                nn.Flatten(start_dim=-3),
            )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1, *input_space))).shape[1]

        self.n_flatten = n_flatten
        self.conv_out_shape = self._conv_out_shape(input_space)
        self.output_dim = feature_dim
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.Mish(),
        )

    def _conv_out_shape(self, input_space):
        '''Spatial shape before flattening. SSL decoders need this to size themselves.'''
        with torch.no_grad():
            x = torch.zeros((1, *input_space))
            for layer in self.cnn:
                if isinstance(layer, nn.Flatten):
                    break
                x = layer(x)
        return tuple(x.shape[1:])

    def forward(self, x):
        return self.fc(self.cnn(x))


def make_encoder(input_space, feature_dim=None):
    if len(input_space) == 1:
        return MLPEncoder(input_space, feature_dim or 64)
    return CNNEncoder(input_space, feature_dim or 256)


class ActorCritic(nn.Module):
    '''
    Encoder + policy head + value head.

    shared_encoder=False reproduces the original PPO's two fully independent networks
    and exists as a regression baseline.

    detach_actor_encoder controls whether policy gradients reach the shared encoder.
    True  -> encoder is shaped by the critic and the SSL task only (CURL / SAC-AE style)
    False -> encoder is shaped by actor, critic and SSL jointly
    Only meaningful when shared_encoder=True.
    '''

    def __init__(
        self,
        observation_space,
        action_space,
        shared_encoder=True,
        detach_actor_encoder=False,
        feature_dim=None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_encoder = shared_encoder
        self.detach_actor_encoder = detach_actor_encoder

        n_actions = sum(action_space)

        if shared_encoder:
            self.encoder = make_encoder(observation_space, feature_dim)
            self.value_encoder = None
            dim = self.encoder.output_dim
        else:
            self.encoder = make_encoder(observation_space, feature_dim)
            self.value_encoder = make_encoder(observation_space, feature_dim)
            dim = self.encoder.output_dim

        self.policy_head = nn.Linear(dim, n_actions)
        self.value_head = nn.Linear(dim, 1)

        self._init_weights()

    def _init_weights(self):
        '''
        Matches the original init: orthogonal on the policy path with the final layer
        scaled down by 100, PyTorch defaults on the value path.

        Note: with a shared encoder the trunk can only be initialised one way, so it
        follows the policy path (orthogonal). This is a deliberate, documented
        deviation from the unshared baseline, where the value encoder got default init.
        '''
        for layer in self.encoder.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        with torch.no_grad():
            nn.init.orthogonal_(self.policy_head.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.policy_head.bias)
            self.policy_head.weight.div_(100)

    @property
    def feature_dim(self):
        return self.encoder.output_dim

    def encode(self, obs):
        '''Shared representation. This is what SSL auxiliary tasks attach to.'''
        return self.encoder(obs)

    def forward(self, obs):
        features = self.encoder(obs)

        if self.shared_encoder:
            policy_features = features.detach() if self.detach_actor_encoder else features
            value_features = features
        else:
            policy_features = features
            value_features = self.value_encoder(obs)

        return self.policy_head(policy_features), self.value_head(value_features), features

    def actor_parameters(self):
        params = list(self.policy_head.parameters())
        if not self.detach_actor_encoder or not self.shared_encoder:
            params += list(self.encoder.parameters())
        return params

    def critic_parameters(self):
        params = list(self.value_head.parameters())
        if self.value_encoder is not None:
            params += list(self.value_encoder.parameters())
        return params
