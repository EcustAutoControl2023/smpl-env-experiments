import numpy as np
import copy
import math
from typing import Any, List, Optional, Sequence, cast

from numpy.lib import mixins

from d3rlpy_patch.models.torch.imitators import TemporalConditionalVAE
import numpy as np
import torch
from torch.optim import Optimizer

from d3rlpy.gpu import Device
from d3rlpy.models.builders import (
    create_conditional_vae,
    create_deterministic_residual_policy,
    create_discrete_imitator,
)
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import (
    ConditionalVAE,
    DeterministicResidualPolicy,
    DiscreteImitator,
    PixelEncoder,
    compute_max_with_n_actions,
)
from ...models.builders import CustomConditionalVAE, create_custom_conditional_vae, create_temporal_conditional_vae
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, torch_api, train_api
from d3rlpy.algos.torch.ddpg_impl import DDPGBaseImpl
from d3rlpy.algos.torch.dqn_impl import DoubleDQNImpl

from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim[0] + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], -1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim[0] + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim[0] + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], -1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], -1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], -1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TBCQImpl(DDPGBaseImpl):

    _imitator_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _beta: float
    _policy: Optional[DeterministicResidualPolicy]
    _targ_policy: Optional[DeterministicResidualPolicy]
    _imitator: Optional[TemporalConditionalVAE]
    _imitator_optim: Optional[Optimizer]

    _k: int
    _tl: int
    _net_type: str
    _num_layers: int

    _history_states: Optional[List[torch.Tensor]]
    _history_actions: Optional[List[Any]]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        lam: float,
        n_action_samples: int,
        action_flexibility: float,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        k: int = 128,
        tl: int = 20,
        net_type: str = "GRU",
        num_layers: int = 1,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._n_critics = n_critics
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._beta = beta

        # initialized in build
        self._imitator = None
        self._imitator_optim = None

        self._k = k
        self._tl = tl
        self._net_type = net_type
        self._num_layers = num_layers

        self._history_states = None
        self._history_actions = None

        self._max_action = 1
        self._phi = 0.05
        self._discount = 0.99
        self._lmbda = 0.75
        self._batch_size = 128


    def build(self) -> None:
        self._build_imitator()
        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

        self._build_actor()
        self._build_critic()

    def _build_actor(self) -> None:
        self.actor = Actor(self._observation_shape, self._action_size, self._max_action, self._phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_learning_rate)

    def _build_critic(self) -> None:
        # critic network
        self.critic = Critic(self._observation_shape, self._action_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_learning_rate)

    def _build_imitator(self) -> None:
        self._imitator = create_temporal_conditional_vae(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            latent_size=2 * self._action_size,
            beta=self._beta,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._imitator_encoder_factory,
            tl=self._tl,
            k=self._k,
            net_type=self._net_type,
            num_layers=self._num_layers,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._imitator is not None

        # reconstruct observation and action
        observation_recon = batch.observations.reshape(-1, self._tl, self.observation_shape[0])
        action_recon = batch.actions.reshape(-1, self._tl, self._action_size)

        observation_recon = observation_recon.permute(1, 0, 2)
        action_recon = action_recon.permute(1, 0, 2)

        sampled_action = self._imitator.predict(
            observation_recon, action_recon[:-1]
        )
        # Update through DPG
        actor_loss = -self.critic.q1(observation_recon[-1], sampled_action).mean()

        return actor_loss

    def update_critic_target(self) -> None:
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    def update_actor_target(self) -> None:
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        # Q function should be inference mode for stability

        self.actor_optimizer.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self.actor_optimizer.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        observation_recon = batch.observations.reshape(-1, self._tl, self.observation_shape[0])
        observation_recon = observation_recon.permute(1, 0, 2)
        action_recon = batch.actions.reshape(-1, self._tl, self._action_size)
        action_recon = action_recon.permute(1, 0, 2)
        current_Q1, current_Q2 = self.critic(observation_recon[-1], action_recon[-1])
        critic_loss = F.mse_loss(current_Q1, q_tpn) + F.mse_loss(current_Q2, q_tpn)
        return critic_loss

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        self.critic_optimizer.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self.critic_optimizer.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api()
    def update_imitator(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._imitator_optim is not None
        assert self._imitator is not None

        self._imitator_optim.zero_grad()

        # reconstruct observation and action
        observation_recon = batch.observations.reshape(-1, self._tl, self.observation_shape[0])
        action_recon = batch.actions.reshape(-1, self._tl, self._action_size)

        observation_recon = observation_recon.permute(1, 0, 2)
        action_recon = action_recon.permute(1, 0, 2)

        # loss = self._imitator.compute_error(batch.observations, batch.actions)
        loss = self._imitator.compute_error(observation_recon, action_recon)

        loss.backward()
        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def _repeat_observation(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self._n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_repeated_action(
        self, repeated_x: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._targ_policy is not None
        # TODO: this seems to be slow with image observation
        # XXX: debug
        # print(f'TBCQImpl._sample_repeated_action: repeated_x.shape={repeated_x.shape}')
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(
            flattened_x.shape[0], 2 * self._action_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        # sampled_action = self._imitator.decode(flattened_x, clipped_latent)
        sampled_action = self._imitator.decode_new(flattened_x)
        # add residual action
        policy = self._targ_policy if target else self._policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self._n_action_samples, self._action_size)

    def _predict_value(
        self,
        repeated_x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        return self._q_func(flattened_x, flattend_action, "none")

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        if self._history_states is None:
            self._history_states = [list(copy.deepcopy(x).detach().cpu().numpy())[0] for _ in range(self._tl)]
            print(f'TBCQImpl._predict_best_action: length of self._history_states={len(self._history_states)}')
        if self._history_actions is None:
            self._history_actions = [list(np.zeros(self._action_size)) for _ in range(self._tl)]
            print(f'TBCQImpl._predict_best_action: length of self._history_actions={len(self._history_actions)}')

        state_tensor = torch.Tensor(self._history_states).unsqueeze(1).to(self._device)
        action_tensor = torch.Tensor(self._history_actions).unsqueeze(1).to(self._device)

        assert self._imitator is not None
        # assert self._policy is not None
        # assert self._q_func is not None

        predicted_action = self._imitator.predict(state_tensor, action_tensor[1:])

        state = x.repeat_interleave(self._n_action_samples, dim=0)
        action = self.actor(
            state,
            predicted_action.repeat_interleave(self._n_action_samples, dim=0)
        )  # repeat batch_size times

        q1 = self.critic.q1(state, action)
        ind = q1.argmax(0)

        # action = action[ind].detach().cpu().numpy().flatten()


        self._history_states.pop(0)
        self._history_states.append(list(copy.deepcopy(x).detach().cpu().numpy())[0])
        self._history_actions.pop(0)
        self._history_actions.append(list(action[ind].detach().cpu().numpy())[0])

        # repeated_x = self._repeat_observation(x)
        # action = self._sample_repeated_action(repeated_x)
        # values = self._predict_value(repeated_x, action)[0]
        # # pick the best (batch_size * n) -> (batch_size,)
        # index = values.view(-1, self._n_action_samples).argmax(dim=1)
        # return action[torch.arange(action.shape[0]), index]
        return action[ind]

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BCQ does not support sampling action")

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._imitator is not None
        # Critic Training
        with torch.no_grad():
            observation_recon = batch.next_observations.reshape(-1, self._tl, self.observation_shape[0])
            observation_recon = observation_recon.permute(1, 0, 2)
            action_recon = batch.actions.reshape(-1, self._tl, self._action_size)
            action_recon = action_recon.permute(1, 0, 2)
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(observation_recon, 10, dim=1)
            history_action = torch.repeat_interleave(action_recon, 10, dim=1)

            target_Q1, target_Q2 = self.critic_target(
                next_state[-1], self.actor_target(next_state[-1], self._imitator.predict(next_state, history_action[1:]))
            )

            # Soft Clipped Double Q-learning
            target_Q = self._lmbda * torch.min(target_Q1, target_Q2) + (1. - self._lmbda) * torch.max(target_Q1, target_Q2)

            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(self._batch_size, -1).max(-1)[0].reshape(-1, 1)
            target_Q = batch.rewards[-1] + (1 - batch.terminals[-1]) * self._discount * target_Q

            return target_Q

class CustomBCQImpl(DDPGBaseImpl):

    _imitator_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _beta: float
    _policy: Optional[DeterministicResidualPolicy]
    _targ_policy: Optional[DeterministicResidualPolicy]
    _imitator: Optional[CustomConditionalVAE]
    _imitator_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        lam: float,
        n_action_samples: int,
        action_flexibility: float,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._n_critics = n_critics
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._beta = beta

        # initialized in build
        self._imitator = None
        self._imitator_optim = None

    def build(self) -> None:
        self._build_imitator()
        super().build()
        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

    def _build_actor(self) -> None:
        self._policy = create_deterministic_residual_policy(
            self._observation_shape,
            self._action_size,
            self._action_flexibility,
            self._actor_encoder_factory,
        )

    def _build_imitator(self) -> None:
        self._imitator = create_custom_conditional_vae(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            latent_size=2 * self._action_size,
            beta=self._beta,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._imitator_encoder_factory,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._q_func is not None
        latent = torch.randn(
            batch.observations.shape[0],
            2 * self._action_size,
            device=self._device,
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self._imitator.decode(
            batch.observations, clipped_latent
        )
        action = self._policy(batch.observations, sampled_action)
        return -self._q_func(batch.observations, action, "none")[0].mean()

    @train_api
    @torch_api()
    def update_imitator(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._imitator_optim is not None
        assert self._imitator is not None

        self._imitator_optim.zero_grad()

        loss = self._imitator.compute_error(batch.observations, batch.actions)

        loss.backward()
        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def _repeat_observation(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self._n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_repeated_action(
        self, repeated_x: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._targ_policy is not None
        # TODO: this seems to be slow with image observation
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(
            flattened_x.shape[0], 2 * self._action_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = self._imitator.decode(flattened_x, clipped_latent)
        # add residual action
        policy = self._targ_policy if target else self._policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self._n_action_samples, self._action_size)

    def _predict_value(
        self,
        repeated_x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        return self._q_func(flattened_x, flattend_action, "none")

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_repeated_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self._n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BCQ does not support sampling action")

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(batch.next_observations)
            actions = self._sample_repeated_action(repeated_x, True)

            values = compute_max_with_n_actions(
                batch.next_observations, actions, self._targ_q_func, self._lam
            )

            return values


class Custom_DiscreteBCQImpl(DoubleDQNImpl):

    _action_flexibility: float
    _beta: float
    _imitator: Optional[DiscreteImitator]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        action_flexibility: float,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )
        self._action_flexibility = action_flexibility
        self._beta = beta

        # initialized in build
        self._imitator = None

    def _build_network(self) -> None:
        super()._build_network()
        assert self._q_func is not None
        # share convolutional layers if observation is pixel
        if isinstance(self._q_func.q_funcs[0].encoder, PixelEncoder):
            self._imitator = DiscreteImitator(
                self._q_func.q_funcs[0].encoder, self._action_size, self._beta
            )
        else:
            self._imitator = create_discrete_imitator(
                self._observation_shape,
                self._action_size,
                self._beta,
                self._encoder_factory,
            )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        assert self._imitator is not None
        q_func_params = list(self._q_func.parameters())
        imitator_params = list(self._imitator.parameters())

        # TODO: replace this with a cleaner way
        # retrieve unique elements
        unique_dict = {}
        for param in q_func_params + imitator_params:
            unique_dict[param] = param
        unique_params = list(unique_dict.values())

        self._optim = self._optim_factory.create(
            unique_params, lr=self._learning_rate
        )

    def compute_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        loss = super().compute_loss(batch, q_tpn)
        imitator_loss = self._imitator.compute_error(
            batch.observations, batch.actions.long()
        )
        return loss + imitator_loss

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._q_func is not None
        log_probs = self._imitator(x)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func(x)
        normalized_value = value - value.min(dim=1, keepdim=True).values
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action
