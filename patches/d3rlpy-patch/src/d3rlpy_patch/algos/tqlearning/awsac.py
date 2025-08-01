from d3rlpy.metrics import evaluate_qlearning_with_environment
import numpy as np
import dataclasses
import math
from typing import Callable, Optional
from d3rlpy.algos.utility import assert_action_space_with_env, build_scalers_with_env
from tqdm.auto import trange
from typing_extensions import Self

from d3rlpy.algos.qlearning.explorers import Explorer
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable, save_config
from d3rlpy.constants import ActionSpace, LoggingStrategy
from d3rlpy.dataset import (
    MixedReplayBuffer,
    ReplayBufferBase,
    create_fifo_replay_buffer,
)
from d3rlpy.logging import LOG, D3RLPyLogger, FileAdapterFactory, LoggerAdapterFactory
from d3rlpy.models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
)
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field
from d3rlpy.optimizers.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.types import GymEnv, Shape
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from .torch.awsac_impl import (
    DiscreteAWSACImpl,
    DiscreteAWSACModules,
    AWSACImpl,
    AWSACModules,
)

__all__ = ["AWSACConfig", "AWSAC", "DiscreteAWSACConfig", "DiscreteAWSAC"]


@dataclasses.dataclass()
class AWSACConfig(LearnableConfig):
    r"""Config Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D,\,
                                   a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} \Big[
            \big(y - Q_{\theta_i}(s_t, a_t)\big)^2\Big]

    .. math::

        y = r_{t+1} + \gamma \Big(\min_j Q_{\theta_j}(s_{t+1}, a_{t+1})
            - \alpha \log \big(\pi_\phi(a_{t+1}|s_{t+1})\big)\Big)

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \Big[\alpha \log (\pi_\phi (a_t|s_t))
              - \min_i Q_{\theta_i}\big(s_t, \pi_\phi(a_t|s_t)\big)\Big]

    The temperature parameter :math:`\alpha` is also automatically adjustable.

    .. math::

        J(\alpha) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \bigg[-\alpha \Big(\log \big(\pi_\phi(a_t|s_t)\big) + H\Big)\bigg]

    where :math:`H` is a target
    entropy, which is defined as :math:`\dim a`.

    References:
        * `Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep
          Reinforcement Learning with a Stochastic Actor.
          <https://arxiv.org/abs/1801.01290>`_
        * `Haarnoja et al., Soft Actor-Critic Algorithms and Applications.
          <https://arxiv.org/abs/1812.05905>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "AWSAC":
        return AWSAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "awsac"


class AWSAC(QLearningAlgoBase[AWSACImpl, AWSACConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
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
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
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
        if self._config.temp_learning_rate > 0:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(),
                lr=self._config.temp_learning_rate,
                compiled=self.compiled,
            )
        else:
            temp_optim = None

        modules = AWSACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = AWSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def fit_online(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBufferBase] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        n_updates: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[GymEnv] = None,
        eval_epsilon: float = 0.0,
        eval_n_trials: int = 10,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        offline_buffer: Optional[ReplayBufferBase] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: Gym-like environment.
            buffer : Replay buffer.
            explorer: Action explorer.
            n_steps: Number of total steps to train.
            n_steps_per_epoch: Number of steps per epoch.
            update_interval: Number of steps per update.
            n_updates: Number of gradient steps at a time. The combination of
                ``update_interval`` and ``n_updates`` controls Update-To-Data
                (UTD) ratio.
            update_start_step: Steps before starting updates.
            random_steps: Steps for the initial random explortion.
            eval_env: Gym-like environment. If None, evaluation is skipped.
            eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
            save_interval: Number of epochs before saving models.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be ``{class name}_online_{timestamp}``.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.
        """

        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)
        # check offline buffer
        assert offline_buffer is not None, "Offline buffer is required."

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        rollout_return = 0.0
        for total_step in xrange(1, n_steps + 1):
            with logger.measure_time("step"):
                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    elif explorer:
                        x = observation.reshape((1,) + observation.shape)
                        action = explorer.sample(self, x, total_step)[0]
                    else:
                        action = self.sample_action(
                            np.expand_dims(observation, axis=0)
                        )[0]

                # step environment
                with logger.measure_time("environment_step"):
                    (
                        next_observation,
                        reward,
                        terminal,
                        truncated,
                        _,
                    ) = env.step(action)
                    rollout_return += float(reward)

                clip_episode = terminal or truncated

                # store observation
                buffer.append(observation, action, float(reward))

                # reset if terminated
                if clip_episode:
                    buffer.clip_episode(terminal)
                    observation, _ = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch

                if (
                    total_step > update_start_step
                    and buffer.transition_count > self.batch_size
                ):
                    if total_step % update_interval == 0:
                        mix_buffer = MixedReplayBuffer(
                            primary_replay_buffer=buffer,
                            secondary_replay_buffer=offline_buffer,
                            secondary_mix_ratio=0.9,
                        )
                        for _ in range(n_updates):  # controls UTD ratio
                            # sample mini-batch
                            with logger.measure_time("sample_batch"):
                                batch = mix_buffer.sample_transition_batch(
                                    self.batch_size
                                )

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss = self.update(batch)

                            # record metrics
                            for name, val in loss.items():
                                logger.add_metric(name, val)

                        if (
                            logging_strategy == LoggingStrategy.STEPS
                            and total_step % logging_steps == 0
                        ):
                            logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_env:
                    eval_score = evaluate_qlearning_with_environment(
                        self,
                        eval_env,
                        n_trials=eval_n_trials,
                        epsilon=eval_epsilon,
                    )
                    logger.add_metric("evaluation", eval_score)

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                if logging_strategy == LoggingStrategy.EPOCH:
                    logger.commit(epoch, total_step)

        # clip the last episode
        buffer.clip_episode(False)

        # close logger
        logger.close()


@dataclasses.dataclass()
class DiscreteAWSACConfig(LearnableConfig):
    r"""Config of Soft Actor-Critic algorithm for discrete action-space.

    This discrete version of SAC is built based on continuous version of SAC
    with additional modifications.

    The target state-value is calculated as expectation of all action-values.

    .. math::

        V(s_t) = \pi_\phi (s_t)^T [Q_\theta(s_t) - \alpha \log (\pi_\phi (s_t))]

    Similarly, the objective function for the temperature parameter is as
    follows.

    .. math::

        J(\alpha) = \pi_\phi (s_t)^T [-\alpha (\log(\pi_\phi (s_t)) + H)]

    Finally, the objective function for the policy function is as follows.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\pi_\phi(s_t)^T [\alpha \log(\pi_\phi(s_t)) - Q_\theta(s_t)]]

    References:
        * `Christodoulou, Soft Actor-Critic for Discrete Action Settings.
          <https://arxiv.org/abs/1910.07207>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 64
    gamma: float = 0.99
    n_critics: int = 2
    initial_temperature: float = 1.0
    target_update_interval: int = 8000

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteAWSAC":
        return DiscreteAWSAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_awsac"


class DiscreteAWSAC(QLearningAlgoBase[DiscreteAWSACImpl, DiscreteAWSACConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        policy = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        if self._config.initial_temperature > 0:
            log_temp = create_parameter(
                (1, 1),
                math.log(self._config.initial_temperature),
                device=self._device,
                enable_ddp=self._enable_ddp,
            )
        else:
            log_temp = None

        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        if self._config.temp_learning_rate > 0:
            assert log_temp is not None
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(),
                lr=self._config.temp_learning_rate,
                compiled=self.compiled,
            )
        else:
            temp_optim = None

        modules = DiscreteAWSACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = DiscreteAWSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(AWSACConfig)
register_learnable(DiscreteAWSACConfig)
