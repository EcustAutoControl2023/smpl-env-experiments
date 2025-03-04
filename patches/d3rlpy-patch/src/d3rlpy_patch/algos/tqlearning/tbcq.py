import dataclasses

from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_deterministic_residual_policy,
    create_discrete_q_function,
    create_vae_decoder,
    create_vae_encoder,
)

from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field
from d3rlpy.models.torch import CategoricalPolicy, compute_output_size
from d3rlpy.optimizers.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.types import Shape

from d3rlpy.algos.qlearning.base import QLearningAlgoBase

from .torch.tbcq_impl import (
    TBCQImpl,
    TBCQModules,
    DiscreteTBCQImpl,
    DiscreteTBCQModules,
)


@dataclasses.dataclass()
class TBCQConfig(LearnableConfig):
    r"""Config of Temporal Batch-Constrained Q-learning algorithm.

    Args:
    observation_scaler (d3rlpy.preprocessing.ObservationScaler):
        Observation preprocessor.
    action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
    reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
    actor_learning_rate (float): Learning rate for policy function.
    critic_learning_rate (float): Learning rate for Q functions.
    imitator_learning_rate (float): Learning rate for Conditional VAE.
    actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
        Optimizer factory for the actor.
    critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
        Optimizer factory for the critic.
    imitator_optim_factory (d3rlpy.optimizers.OptimizerFactory):
        Optimizer factory for the conditional VAE.
    actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
        Encoder factory for the actor.
    critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
        Encoder factory for the critic.
    imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
        Encoder factory for the conditional VAE.
    q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
        Q function factory.
    batch_size (int): Mini-batch size.
    gamma (float): Discount factor.
    tau (float): Target network synchronization coefficiency.
    n_critics (int): Number of Q functions for ensemble.
    update_actor_interval (int): Interval to update policy function.
    lam (float): Weight factor for critic ensemble.
    n_action_samples (int): Number of action samples to estimate
        action-values.
    action_flexibility (float): Output scale of perturbation function
        represented as :math:`\Phi`.
    rl_start_step (int): Steps to start to update policy function and Q
        functions. If this is large, RL training would be more stabilized.
    beta (float): KL reguralization term for Conditional VAE.
    compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    imitator_learning_rate: float = 1e-3
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    imitator_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    imitator_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    update_actor_interval: int = 1
    lam: float = 0.75
    n_action_samples: int = 100
    action_flexibility: float = 0.05
    rl_start_step: int = 0
    beta: float = 0.5

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "TBCQ":
        return TBCQ(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "tbcq"


class TBCQ(QLearningAlgoBase[TBCQImpl, TBCQConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        policy = create_deterministic_residual_policy(
            observation_shape,
            action_size,
            self._config.action_flexibility,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_policy = create_deterministic_residual_policy(
            observation_shape,
            action_size,
            self._config.action_flexibility,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        vae_encoder = create_vae_encoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        vae_decoder = create_vae_decoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        vae_optim = self._config.imitator_optim_factory.create(
            list(vae_encoder.named_modules()) + list(vae_decoder.named_modules()),
            lr=self._config.imitator_learning_rate,
            compiled=self.compiled,
        )

        modules = TBCQModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            vae_optim=vae_optim,
        )

        self._impl = TBCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            lam=self._config.lam,
            n_action_samples=self._config.n_action_samples,
            action_flexibility=self._config.action_flexibility,
            beta=self._config.beta,
            rl_start_step=self._config.rl_start_step,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteTBCQConfig(LearnableConfig):
    r"""Config of Discrete version of Batch-Constrained Q-learning algorithm.

    Discrete version takes theories from the continuous version, but the
    algorithm is much simpler than that.
    The imitation function :math:`G_\omega(a|s)` is trained as supervised
    learning just like Behavior Cloning.

    .. math::

        L(\omega) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log G_\omega(a|s_t)]

    With this imitation function, the greedy policy is defined as follows.

    .. math::

        \pi(s_t) = \text{argmax}_{a|G_\omega(a|s_t)
                / \max_{\tilde{a}} G_\omega(\tilde{a}|s_t) > \tau}
            Q_\theta (s_t, a)

    which eliminates actions with probabilities :math:`\tau` times smaller
    than the maximum one.

    Finally, the loss function is computed in Double DQN style with the above
    constrained policy.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\theta'}(s_{t+1}, \pi(s_{t+1}))
            - Q_\theta(s_t, a_t))^2]

    References:
        * `Fujimoto et al., Off-Policy Deep Reinforcement Learning without
          Exploration. <https://arxiv.org/abs/1812.02900>`_
        * `Fujimoto et al., Benchmarking Batch Deep Reinforcement Learning
          Algorithms. <https://arxiv.org/abs/1910.01708>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        learning_rate (float): Learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            Encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        action_flexibility (float): Probability threshold represented as
            :math:`\tau`.
        beta (float): Reguralization term for imitation function.
        target_update_interval (int): Interval to update the target network.
        share_encoder (bool): Flag to share encoder between Q-function and
            imitation models.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 32
    gamma: float = 0.99
    n_critics: int = 1
    action_flexibility: float = 0.3
    beta: float = 0.5
    target_update_interval: int = 8000
    share_encoder: bool = True

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteTBCQ":
        return DiscreteTBCQ(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_tbcq"


class DiscreteTBCQ(QLearningAlgoBase[DiscreteTBCQImpl, DiscreteTBCQConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        # share convolutional layers if observation is pixel
        if self._config.share_encoder:
            hidden_size = compute_output_size(
                [observation_shape],
                q_funcs[0].encoder,
            )
            imitator = CategoricalPolicy(
                encoder=q_funcs[0].encoder,
                hidden_size=hidden_size,
                action_size=action_size,
            )
            imitator.to(self._device)
        else:
            imitator = create_categorical_policy(
                observation_shape,
                action_size,
                self._config.encoder_factory,
                device=self._device,
                enable_ddp=self._enable_ddp,
            )

        q_func_params = list(q_funcs.named_modules())
        imitator_params = list(imitator.named_modules())
        optim = self._config.optim_factory.create(
            q_func_params + imitator_params,
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteTBCQModules(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            imitator=imitator,
            optim=optim,
        )

        self._impl = DiscreteTBCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            action_flexibility=self._config.action_flexibility,
            beta=self._config.beta,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(TBCQConfig)
register_learnable(DiscreteTBCQConfig)
