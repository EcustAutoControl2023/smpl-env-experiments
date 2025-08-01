import dataclasses
import math
from queue import Queue
from typing import Optional, Callable
from d3rlpy.algos.utility import assert_action_space_with_env, build_scalers_with_env
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.metrics import evaluate_qlearning_with_environment
import torch
from torch import exp, sqrt
from d3rlpy_patch.algos.experts import Expert
from tqdm.std import trange
from typing_extensions import Self
import numpy as np

from d3rlpy.base import (
    LOG,
    D3RLPyLogger,
    DeviceArg,
    LearnableConfig,
    register_learnable,
    save_config,
)
from d3rlpy.constants import ActionSpace, LoggingStrategy
from d3rlpy.dataset import (
    MixedReplayBuffer,
    ReplayBufferBase,
    create_fifo_replay_buffer,
)
from d3rlpy.logging import FileAdapter, FileAdapterFactory, LoggerAdapterFactory
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
from d3rlpy.types import Shape
from d3rlpy.algos.qlearning.base import Explorer, QLearningAlgoBase
from .torch.sacif_impl import (
    DiscreteSACIFImpl,
    DiscreteSACIFModules,
    SACIFImpl,
    SACIFModules,
)

from d3rlpy.types import GymEnv

__all__ = ["SACIFConfig", "SACIF", "DiscreteSACIFConfig", "DiscreteIFSAC"]


@dataclasses.dataclass()
class SACIFConfig(LearnableConfig):
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

    ema_beta: float = 0.995

    intervention: bool = False
    intervention_method: str = "Constant"
    intervention_rate: float = 0.5  # grid search parameter {0.25, 0.5, 0.75, 1.0}
    intervention_length: int = 3
    intervention_degree: float = 0.5
    intervention_k: int = 3
    intervention_T: int = 1

    intervention_stop_step: int = 0
    intervention_degree_start: float = 0.5
    intervention_degree_end: float = 0.5

    # FIXME: mixing_ratio is not used in SACIF
    # XXX: for compatitive
    mixing_ratio: float = 0.5

    buffer_method: str = "Constant"
    buffer_start: float = 0.5
    buffer_end: float = 0.5
    buffer_k: int = 3

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "SACIF":
        return SACIF(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "sac_if"


class SACIF(QLearningAlgoBase[SACIFImpl, SACIFConfig], QLearningAlgoProtocol):
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

        modules = SACIFModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = SACIFImpl(
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
        expert: Optional[Expert] = None,
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

        print("overwrite fit_online !!")
        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)
        # check offline buffer
        assert offline_buffer is not None, "offline_buffer is None"

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

        observation_queue = []
        action_queue = []
        reward_queue = []

        # ema rollout return
        rollout_return_ema = 0.0
        # ema evaluation return
        eval_return_ema = 0.0
        # 平滑化の係数
        alpha = 0.99

        # start training loop
        observation, _ = env.reset()
        rollout_return = 0.0
        prev_intervention = False

        error_num = 0
        running_var_q = 0
        action = None

        for total_step in xrange(1, n_steps + 1):
            with logger.measure_time("step"):
                # timestep in the current epoch
                timestep = total_step % n_steps_per_epoch
                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    elif expert:
                        if self._config.intervention_method == "Constant":
                            self._config.intervention_degree = (
                                self._config.intervention_degree_start
                            )
                        elif self._config.intervention_method == "Linear":
                            self._config.intervention_degree = (
                                self._config.intervention_degree_start
                                + (
                                    self._config.intervention_degree_end
                                    - self._config.intervention_degree_start
                                )
                                * (
                                    timestep
                                    / (self._config.intervention_T * n_steps_per_epoch)
                                )
                            )
                        elif self._config.intervention_method == "Exponential":
                            self._config.intervention_degree = np.exp(
                                -1
                                * self._config.intervention_T
                                * (timestep / n_steps_per_epoch)
                            )
                        elif self._config.intervention_method == "ConfidenceBased":
                            # Compute uncertainty U_A(s_t, a_t)
                            # add demension for batch processing
                            if action is None:
                                # if action is None, keep the default intervention degree
                                pass
                            else:
                                batch_observation = np.expand_dims(observation, axis=0)
                                batch_action = np.expand_dims(action, axis=0)
                                # convert to torch tensor
                                batch_observation = torch.tensor(
                                    batch_observation,
                                    dtype=torch.float32,
                                    device=self._device,
                                )
                                batch_action = torch.tensor(
                                    batch_action,
                                    dtype=torch.float32,
                                    device=self._device,
                                )
                                assert self._impl is not None, (
                                    "SACIFImpl is not initialized"
                                )
                                mean, var = self._impl.compute_mean_variance(
                                    batch_observation, batch_action
                                )
                                # update running EMA of variance
                                running_var_q = (
                                    self._config.ema_beta * running_var_q
                                    + (1 - self._config.ema_beta) * var
                                    + 1e-8
                                )
                                u_a = sqrt(var / running_var_q)
                                # U_A save to logger
                                self._config.intervention_degree = float(
                                    self._config.intervention_degree_start
                                    / (
                                        1
                                        + exp(-self._config.intervention_k * (u_a - 1))
                                    )
                                )

                                logger.add_metric("U_A", u_a.item())
                        else:
                            raise ValueError(
                                f"Unknown intervention method: {self._config.intervention_method}"
                            )

                        logger.add_metric(
                            "intervention_degree",
                            self._config.intervention_degree,
                        )
                        assert isinstance(logger._adapter, FileAdapter), (
                            "Logger adapter is not FileAdapter"
                        )
                        file_path = (
                            logger._adapter._logdir
                            + f"/intervention_degree_{n_steps_per_epoch * (total_step // n_steps_per_epoch + 1)}.csv"
                        )
                        with open(file_path, "a") as f:
                            f.write(f"{timestep},{self._config.intervention_degree}\n")

                        if not self._config.intervention:
                            self._config.intervention = bool(
                                np.random.choice(
                                    a=[0, 1],
                                    size=1,
                                    p=[
                                        1 - self._config.intervention_rate,
                                        self._config.intervention_rate,
                                    ],
                                )
                            )
                            if self._config.intervention:
                                self._config.intervention_stop_step = (
                                    timestep + self._config.intervention_length
                                )
                        if (
                            self._config.intervention
                            and timestep == self._config.intervention_stop_step
                        ):
                            self._config.intervention = False

                        x = observation.reshape((1,) + observation.shape)
                        if self._config.intervention:
                            action_infer = self.sample_action(
                                np.expand_dims(observation, axis=0)
                            )[0]
                            action = (
                                self._config.intervention_degree
                                * expert.guide(self, x, timestep)
                                + (1 - self._config.intervention_degree) * action_infer
                            )
                        else:
                            if explorer:
                                x = observation.reshape((1,) + observation.shape)
                                action = explorer.sample(self, x, total_step)[0]
                            else:
                                action = self.sample_action(
                                    np.expand_dims(observation, axis=0)
                                )[0]
                    elif explorer:
                        x = observation.reshape((1,) + observation.shape)
                        action = explorer.sample(self, x, total_step)[0]
                    else:
                        action = self.sample_action(
                            np.expand_dims(observation, axis=0)
                        )[0]

                if self._config.intervention:
                    if timestep != 1 and not prev_intervention:
                        buffer.append(
                            observation_queue[-1],
                            action_queue[-1],
                            reward_queue[-1] - self._config.intervention_degree,
                        )
                    else:
                        if timestep != 1:
                            buffer.append(
                                observation_queue[-1],
                                action_queue[-1],
                                reward_queue[-1],
                            )
                else:
                    if timestep != 1:
                        buffer.append(
                            observation_queue[-1],
                            action_queue[-1],
                            reward_queue[-1],
                        )

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

                if timestep == 1:
                    buffer.append(observation, action, float(reward))

                clip_episode = terminal or truncated

                # append to queues
                prev_intervention = self._config.intervention
                observation_queue.append(observation)
                action_queue.append(action)
                reward_queue.append(float(reward))

                # reset if terminated
                if clip_episode:
                    buffer.clip_episode(terminal)
                    if _.get("error_occurred", False):
                        error_num += 1
                    observation, _ = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    if rollout_return_ema == 0.0:
                        rollout_return_ema = rollout_return
                    rollout_return_ema = (
                        alpha * rollout_return_ema + (1 - alpha) * rollout_return
                    )
                    logger.add_metric("rollout_return_ema", rollout_return_ema)
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
                        mix_ratio = None
                        if self._config.buffer_method == "Constant":
                            mix_ratio = self._config.buffer_start
                        elif self._config.buffer_method == "Linear":
                            mix_ratio = self._config.buffer_start + (
                                self._config.buffer_end - self._config.buffer_start
                            ) * (timestep / (self._config.buffer_k * n_steps_per_epoch))
                        elif self._config.buffer_method == "Exponential":
                            mix_ratio = np.exp(
                                -1
                                * self._config.buffer_k
                                * (timestep / n_steps_per_epoch)
                            )
                        else:
                            raise ValueError(
                                f"Unknown buffer method: {self._config.buffer_method}"
                            )

                        assert mix_ratio is not None, "mix_ratio is None"
                        mix_buffer = MixedReplayBuffer(
                            primary_replay_buffer=buffer,
                            secondary_replay_buffer=offline_buffer,
                            secondary_mix_ratio=mix_ratio,
                        )

                        for _ in range(n_updates):  # controls UTD ratio
                            # sample mini-batch
                            with logger.measure_time("sample_batch"):
                                # batch = buffer.sample_transition_batch(self.batch_size)
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
                    if eval_return_ema == 0.0:
                        eval_return_ema = eval_score
                    eval_return_ema = alpha * eval_return_ema + (1 - alpha) * eval_score
                    logger.add_metric("evaluation_ema", eval_return_ema)

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                if logging_strategy == LoggingStrategy.EPOCH:
                    logger.commit(epoch, total_step)

                # clear queue
                observation_queue = []
                action_queue = []
                reward_queue = []
                prev_intervention = False
                self._config.intervention = False
                observation, _ = env.reset()
                logger.add_metric("error_occurred", error_num)
                error_num = 0

        # clip the last episode
        buffer.clip_episode(False)

        # close logger
        logger.close()


@dataclasses.dataclass()
class DiscreteSACIFConfig(LearnableConfig):
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
    ) -> "DiscreteIFSAC":
        return DiscreteIFSAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_sacif"


class DiscreteIFSAC(QLearningAlgoBase[DiscreteSACIFImpl, DiscreteSACIFConfig]):
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

        modules = DiscreteSACIFModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = DiscreteSACIFImpl(
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


register_learnable(SACIFConfig)
register_learnable(DiscreteSACIFConfig)
