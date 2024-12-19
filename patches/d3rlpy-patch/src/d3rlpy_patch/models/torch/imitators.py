from typing import Sequence, cast, Optional, Dict
from typing import Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence
from d3rlpy.models.torch import Encoder, EncoderWithAction

from d3rlpy_patch.models.torch.func import multivariate_normal_kl_loss, normal_differential_sample, zeros_like_with_shape

from .common import PreProcess, DBlock, logsigma2cov


class TemporalConditionalVAE(nn.Module):  # type: ignore
    _encoder_encoder: EncoderWithAction
    _decoder_encoder: EncoderWithAction
    _beta: float
    _min_logstd: float
    _max_logstd: float

    _action_size: int
    _latent_size: int
    _mu: nn.Linear
    _logstd: nn.Linear
    _fc: nn.Linear

    _k: int # 对应TBCQ中的batch_size（self.k）
    _observation_size: int
    _num_layers: int # 对应TBCQ中的num_layers（RNN or GRU的层数）
    _net_type: str # 对应TBCQ中的net_type（RNN or GRU）

    _rnn: nn.Module         # 对应TBCQ中的rnn（RNN or GRU）
    _process_u: nn.Module   # 对应TBCQ中的process_u（Preprocess)
    _process_x: nn.Module   # 对应TBCQ中的process_x（Preprocess)
    _process_z: nn.Module   # 对应TBCQ中的process_z（Preprocess)
    _process_e: PreProcess   # 对应TBCQ中的process_e（Preprocess)

    _posterior_gaussian: nn.Module # 对应TBCQ中的posterior_gaussian（DBlock）
    _prior_gaussian: nn.Module     # 对应TBCQ中的prior_gaussian（DBlock）
    _decoder: nn.Module     # 对应TBCQ中的decoder（DBlock）

    _tl: int # 对应TBCQ中的时间长度（self.tl）

    def __init__(
        self,
        encoder_encoder: EncoderWithAction,
        decoder_encoder: EncoderWithAction,
        beta: float,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
        tl: int = 10,
        k: int = 128,
        net_type: str = 'GRU',
        num_layers: int = 1,
    ):
        super().__init__()
        self._encoder_encoder = encoder_encoder
        self._decoder_encoder = decoder_encoder
        self._beta = beta
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd

        self._action_size = encoder_encoder.action_size
        self._latent_size = decoder_encoder.action_size

        # encoder
        self._mu = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        self._logstd = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        # decoder
        self._fc = nn.Linear(
            decoder_encoder.get_feature_size(), self._action_size
        )

        #### TBCQ中的部分 ####
        self._k = k
        self._observation_size = encoder_encoder.observation_shape[0]
        self._num_layers = num_layers
        if net_type == 'RNN':
            RNNClass = nn.RNN
        elif net_type == 'GRU':
            RNNClass = nn.GRU
        else:
            raise NotImplementedError(f'net_type {net_type} is not supported.')

        self._rnn = RNNClass(3*self._k, self._k, num_layers=self._num_layers)
        self._process_u = PreProcess(self._action_size, self._k)
        self._process_x = PreProcess(self._observation_size, self._k)
        self._process_z = PreProcess(self._latent_size, self._k)
        self._process_e = PreProcess(self._latent_size, self._k)

        self._posterior_gaussian = DBlock(2*self._k, 3*self._k, self._latent_size)
        self._prior_gaussian = DBlock(self._k, 3*self._k, self._latent_size)
        self._decoder = DBlock(2*self._k, 3*self._k, self._observation_size)

        self._e1 = nn.Linear(4*self._k, 750)
        self._e2 = nn.Linear(750, 750)
        self._mean = nn.Linear(750, self._latent_size)
        self._logstd = nn.Linear(750, self._latent_size)
        self._d1 = nn.Linear(4*self._k, 750)
        self._d2 = nn.Linear(750, 750)
        self._d3 = nn.Linear(750, self._action_size)

        self._tl = tl

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def forward_posterior(self, x: torch.Tensor, action: torch.Tensor, memory_state: Optional[Dict]=None) -> Tuple[Dict, Dict]:
        x_seq_embed = self._process_x(x)
        action_seq_embed = self._process_u(action)

        l, batch_size, _ = action.shape

        h, rnn_hidden_state = (
            zeros_like_with_shape(action, (batch_size, self._k)),
            zeros_like_with_shape(action, (self._num_layers, batch_size, self._k)),
        ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden'])

        z_t = zeros_like_with_shape(action, (batch_size, self._k))

        state_mu_list = []
        state_logstd_list = []
        sampled_state_list = []
        h_list = [h]
        rnn_hidden_state_list = [rnn_hidden_state.transpose(1, 0)]

        for t in range(self._tl):
            z_t_mean, z_t_logstd = self._posterior_gaussian(torch.cat([x_seq_embed[t], h], dim=-1))
            z_t = normal_differential_sample(MultivariateNormal(z_t_mean, logsigma2cov(z_t_logstd)))

            output, rnn_hidden_state = self._rnn(torch.cat(
                [x_seq_embed[t], action_seq_embed[t], self._process_z(z_t)], dim=-1).unsqueeze(dim=0), rnn_hidden_state
            )
            h = output[0]

            state_mu_list.append(z_t_mean)
            state_logstd_list.append(z_t_logstd)
            sampled_state_list.append(z_t)
            h_list.append(h)
            rnn_hidden_state_list.append(rnn_hidden_state.contiguous().transpose(1, 0))

        state_mu = torch.stack(state_mu_list, dim=0)
        state_logstd = torch.stack(state_logstd_list, dim=0)
        sampled_state = torch.stack(sampled_state_list, dim=0)
        h_seq = torch.stack(h_list, dim=0)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_list, dim=0)

        z_t_embed = self._process_z(z_t)
        e_t = F.relu(self._e1(torch.cat([h_seq[-2], x_seq_embed[-1], action_seq_embed[-1], z_t_embed], dim=-1)))
        e_t = F.relu(self._e2(e_t))
        e_t_mean = self._mean(e_t)
        e_t_std = self._logstd(e_t).clamp(self._min_logstd, self._max_logstd)
        e_t = e_t_mean + e_t_std * torch.randn_like(e_t_std)
        a_t = self.decode_action(h_seq[-2], e_t, x_seq_embed[-1], z_t_embed)

        # XXX: debug
        # print('a_t:', a_t.shape)
        # print('e_t:', e_t.shape)
        # print('e_t_mean:', e_t_mean.shape)
        # print('e_t_std:', e_t_std.shape)
        # print('state_mu:', state_mu.shape)
        # print('state_logstd:', state_logstd.shape)
        # print('sampled_state:', sampled_state.shape)
        # print('h_seq:', h_seq.shape)
        # print('action_seq_embed:', action_seq_embed.shape)
        # print('rnn_hidden_state_seq:', rnn_hidden_state_seq.shape)

        outputs = {
            'a_t': a_t,
            'e_t': e_t,
            'e_t_mean': e_t_mean,
            'e_t_std': e_t_std,
            'state_mu': state_mu,
            'state_logstd': state_logstd,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'action_seq_embed': action_seq_embed,
            'rnn_hidden_state_seq': rnn_hidden_state_seq,
        }

        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def call_loss(self, action_seq, x_seq, memory_state=None):
        outputs, memory_state = self.forward_posterior(x_seq, action_seq, memory_state)

        h_seq = outputs['h_seq']
        state_mu = outputs['state_mu']
        state_logstd = outputs['state_logstd']
        a_t_recon = outputs['a_t']
        e_t_mean = outputs['e_t_mean']
        e_t_std = outputs['e_t_std']

        l, batch_size, _ = x_seq.shape

        predicted_h = h_seq[-1]

        prior_z_t_seq_mean, prior_z_t_seq_logstd = self._prior_gaussian(predicted_h)

        z_kl_loss = multivariate_normal_kl_loss(state_mu, logsigma2cov(state_logstd), prior_z_t_seq_mean, logsigma2cov(prior_z_t_seq_logstd))

        observation_normal_sample = self.decode_observation(outputs, mode='sample')
        x_recon_loss = F.mse_loss(observation_normal_sample, x_seq)

        vrnn_loss = z_kl_loss + x_recon_loss

        a_recon_loss = F.mse_loss(a_t_recon, action_seq[-1])
        e_kl_loss = -0.5 * (1+torch.log(e_t_std**2) - e_t_mean**2 - e_t_std**2).mean()
        vanilla_vae_loss = a_recon_loss + e_kl_loss

        return vrnn_loss, vanilla_vae_loss, a_recon_loss

    def decode_observation(self, outputs, mode='sample'):
        mean, logstd = self._decoder(torch.cat([self._process_z(outputs['sampled_state']), outputs['h_seq'][:-1]], dim=-1))

        observation_normal = MultivariateNormal(mean, logsigma2cov(logstd))

        if mode == 'sample':
            return observation_normal.sample()
        elif mode == 'dist':
            return observation_normal

    def decode_action(self, h: torch.Tensor, e: Optional[torch.Tensor], x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if e is None:
            e = torch.randn((x.shape[0], self._latent_size), device=x.device).clamp(-0.5, 0.5)

        e = self._process_e.forward(e)
        d = F.relu(self._d1(torch.cat([h, e, x, z], dim=-1)))
        d = F.relu(self._d2(d))
        # FIXME: self._max_action is not defined
        return torch.tanh(self._d3(d))

    def predict(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        external_input_seq = action
        # FIXME: 不知道原论文这里是什么意思？？
        observation_seq = x # x[:-1]

        outputs, _ = self.forward_posterior(observation_seq, external_input_seq)

        h_seq = outputs['h_seq']
        sampled_state = outputs['sampled_state']
        e_t = outputs['e_t']

        z_t_embed = self._process_z(sampled_state[-1])
        x_seq_embed = self._process_x(x)

        a_t = self.decode_action(h_seq[-2], e_t, x_seq_embed[-1], z_t_embed)

        return a_t

    def encode(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
        h = self._encoder_encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return torch.tanh(self._fc(h))
        # return self.predict(x, latent)

    def decode_new(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample(x)

    def decode_without_squash(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return self._fc(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # dist = self.encode(x, action)
        # kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        # y = self.decode(x, dist.rsample())
        # return F.mse_loss(y, action) + cast(torch.Tensor, self._beta * kl_loss)
        vrnn_loss, vanilla_vae_loss, a_recon_loss = self.call_loss(action, x)
        return vrnn_loss + vanilla_vae_loss

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        # latent = torch.randn((x.shape[0], self._latent_size), device=x.device)
        # # to prevent extreme numbers
        # return self.decode(x, latent.clamp(-0.5, 0.5))
        h = torch.randn((x.shape[0], self._k), device=x.device)
        e_t = torch.randn((x.shape[0], self._latent_size), device=x.device)
        x_seq_embed = self._process_x(x)
        z_t_mean, z_t_logstd = self._posterior_gaussian(torch.cat([x_seq_embed, h], dim=-1))
        z_t = normal_differential_sample(MultivariateNormal(z_t_mean, logsigma2cov(z_t_logstd)))
        z_t_embed = self._process_z(z_t)

        return self.decode_action(h, e_t, x_seq_embed, z_t_embed)

    def sample_n(
        self, x: torch.Tensor, n: int, with_squash: bool = True
    ) -> torch.Tensor:
        flat_latent_shape = (n * x.shape[0], self._latent_size)
        flat_latent = torch.randn(flat_latent_shape, device=x.device)
        # to prevent extreme numbers
        clipped_latent = flat_latent.clamp(-0.5, 0.5)

        # (batch, obs) -> (n, batch, obs)
        repeated_x = x.expand((n, *x.shape))
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = repeated_x.reshape(-1, *x.shape[1:])

        if with_squash:
            flat_actions = self.decode(flat_x, clipped_latent)
        else:
            flat_actions = self.decode_without_squash(flat_x, clipped_latent)

        # (n * batch, action) -> (n, batch, action)
        actions = flat_actions.view(n, x.shape[0], -1)

        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n(x, n, with_squash=False)

class CustomConditionalVAE(nn.Module):  # type: ignore
    _encoder_encoder: EncoderWithAction
    _decoder_encoder: EncoderWithAction
    _beta: float
    _min_logstd: float
    _max_logstd: float

    _action_size: int
    _latent_size: int
    _mu: nn.Linear
    _logstd: nn.Linear
    _fc: nn.Linear

    _k: int # 对应TBCQ中的batch_size（self.k）
    _observation_size: int
    _num_layers: int # 对应TBCQ中的num_layers（RNN or GRU的层数）
    _net_type: str # 对应TBCQ中的net_type（RNN or GRU）

    _rnn: nn.Module         # 对应TBCQ中的rnn（RNN or GRU）
    _process_u: nn.Module   # 对应TBCQ中的process_u（Preprocess)
    _process_x: nn.Module   # 对应TBCQ中的process_x（Preprocess)
    _process_z: nn.Module   # 对应TBCQ中的process_z（Preprocess)
    _process_e: nn.Module   # 对应TBCQ中的process_e（Preprocess)

    _posterior_gaussian: nn.Module # 对应TBCQ中的posterior_gaussian（DBlock）
    _prior_gaussian: nn.Module     # 对应TBCQ中的prior_gaussian（DBlock）



    def __init__(
        self,
        encoder_encoder: EncoderWithAction,
        decoder_encoder: EncoderWithAction,
        beta: float,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self._encoder_encoder = encoder_encoder
        self._decoder_encoder = decoder_encoder
        self._beta = beta
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd

        self._action_size = encoder_encoder.action_size
        self._latent_size = decoder_encoder.action_size

        # encoder
        self._mu = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        self._logstd = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        # decoder
        self._fc = nn.Linear(
            decoder_encoder.get_feature_size(), self._action_size
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def forward_posterior(self, x: torch.Tensor, action: torch.Tensor, memory_state: Optional[Dict]=None) -> None:
        x_seq_embed = self._process_x(x)
        action_seq_embed = self._process_u(action)

        l, batch_size, _ = action.shape

        h, rnn_hidden_state = (
            zeros_like_with_shape(action, (batch_size, self._k)),
            zeros_like_with_shape(action, (self._num_layers, batch_size, self._k)),
        ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden_state'])

    def encode(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
        h = self._encoder_encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return torch.tanh(self._fc(h))

    def decode_without_squash(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return self._fc(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + cast(torch.Tensor, self._beta * kl_loss)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.randn((x.shape[0], self._latent_size), device=x.device)
        # to prevent extreme numbers
        return self.decode(x, latent.clamp(-0.5, 0.5))

    def sample_n(
        self, x: torch.Tensor, n: int, with_squash: bool = True
    ) -> torch.Tensor:
        flat_latent_shape = (n * x.shape[0], self._latent_size)
        flat_latent = torch.randn(flat_latent_shape, device=x.device)
        # to prevent extreme numbers
        clipped_latent = flat_latent.clamp(-0.5, 0.5)

        # (batch, obs) -> (n, batch, obs)
        repeated_x = x.expand((n, *x.shape))
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = repeated_x.reshape(-1, *x.shape[1:])

        if with_squash:
            flat_actions = self.decode(flat_x, clipped_latent)
        else:
            flat_actions = self.decode_without_squash(flat_x, clipped_latent)

        # (n * batch, action) -> (n, batch, action)
        actions = flat_actions.view(n, x.shape[0], -1)

        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n(x, n, with_squash=False)


class OriginalConditionalVAE(nn.Module):  # type: ignore
    _encoder_encoder: EncoderWithAction
    _decoder_encoder: EncoderWithAction
    _beta: float
    _min_logstd: float
    _max_logstd: float

    _action_size: int
    _latent_size: int
    _mu: nn.Linear
    _logstd: nn.Linear
    _fc: nn.Linear

    def __init__(
        self,
        encoder_encoder: EncoderWithAction,
        decoder_encoder: EncoderWithAction,
        beta: float,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self._encoder_encoder = encoder_encoder
        self._decoder_encoder = decoder_encoder
        self._beta = beta
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd

        self._action_size = encoder_encoder.action_size
        self._latent_size = decoder_encoder.action_size

        # encoder
        self._mu = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        self._logstd = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        # decoder
        self._fc = nn.Linear(
            decoder_encoder.get_feature_size(), self._action_size
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def encode(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
        h = self._encoder_encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return torch.tanh(self._fc(h))

    def decode_without_squash(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return self._fc(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + cast(torch.Tensor, self._beta * kl_loss)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.randn((x.shape[0], self._latent_size), device=x.device)
        # to prevent extreme numbers
        return self.decode(x, latent.clamp(-0.5, 0.5))

    def sample_n(
        self, x: torch.Tensor, n: int, with_squash: bool = True
    ) -> torch.Tensor:
        flat_latent_shape = (n * x.shape[0], self._latent_size)
        flat_latent = torch.randn(flat_latent_shape, device=x.device)
        # to prevent extreme numbers
        clipped_latent = flat_latent.clamp(-0.5, 0.5)

        # (batch, obs) -> (n, batch, obs)
        repeated_x = x.expand((n, *x.shape))
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = repeated_x.reshape(-1, *x.shape[1:])

        if with_squash:
            flat_actions = self.decode(flat_x, clipped_latent)
        else:
            flat_actions = self.decode_without_squash(flat_x, clipped_latent)

        # (n * batch, action) -> (n, batch, action)
        actions = flat_actions.view(n, x.shape[0], -1)

        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n(x, n, with_squash=False)
