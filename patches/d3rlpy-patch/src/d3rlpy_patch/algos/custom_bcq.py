from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Union, Tuple, cast

from d3rlpy.iterators import RandomIterator, TransitionIterator
from d3rlpy.logger import LOG, D3RLPyLogger
from d3rlpy_patch.iterators.round_iterator import RoundIterator
from d3rlpy_patch.models.torch.imitators import TemporalConditionalVAE
import numpy as np

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from d3rlpy.constants import CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR, DISCRETE_ACTION_SPACE_MISMATCH_ERROR, IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from tqdm import tqdm
from .torch.custom_bcq_impl import TBCQMImpl,  TBCQImpl


class TBCQ(AlgoBase):
    '''
    TBCQ is the very first practical data-driven deep reinforcement learning
    '''
    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _rl_start_step: int
    _beta: float
    _use_gpu: Optional[Device]
    _impl: Optional[TBCQImpl]

    _k: int
    _tl: int
    _net_type: str
    _num_layers: int

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        imitator_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        lam: float = 0.75,
        n_action_samples: int = 20,
        action_flexibility: float = 0.05,
        rl_start_step: int = 0,
        beta: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[TBCQImpl] = None,
        k: int = 128,
        tl: int = 20,
        net_type: str = "GRU",
        num_layers: int = 1,
        **kwargs: Any
    ):
        print('Implimentation of TBCQ!')
        print(f'batch_size: {batch_size}')
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._imitator_learning_rate = imitator_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_actor_interval = update_actor_interval
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._rl_start_step = rl_start_step
        self._beta = beta
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._k = k
        self._tl = tl
        self._net_type = net_type
        self._num_layers = num_layers

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = TBCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            action_flexibility=self._action_flexibility,
            beta=self._beta,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            k=self._k,
            tl=self._tl,
            net_type=self._net_type,
            num_layers=self._num_layers
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # XXX: debug
        # print(f'batch n_steps: {batch.n_steps}')
        # __import__('pprint').pprint(f'batch observations: {batch.observations.shape}')
        # print(f'observation shape: {self.observation_shape}')

        imitator_loss = self._impl.update_imitator(batch)
        metrics.update({"imitator_loss": imitator_loss})

        if self._grad_step >= self._rl_start_step:
            critic_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})

            if self._grad_step % self._update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch)
                metrics.update({"actor_loss": actor_loss})
                self._impl.update_actor_target()
                self._impl.update_critic_target()

        return metrics

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """BCQ does not support sampling action."""
        raise NotImplementedError("BCQ does not support sampling action.")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def _evaluate(self, episodes: List[Episode], scorers: Dict[str, Callable[[Any, List[Episode]], float]], logger: D3RLPyLogger) -> None:
        # __import__('pprint').pprint(scorers)
        return super()._evaluate(episodes, scorers, logger)

    def fitter(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: offline dataset to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            iterator yielding current epoch and metrics dict.

        """

        transitions = []
        if isinstance(dataset, MDPDataset):
            for episode in dataset.episodes:
                transitions += episode.transitions
        elif not dataset:
            raise ValueError("empty dataset is not supported.")
        elif isinstance(dataset[0], Episode):
            for episode in cast(List[Episode], dataset):
                transitions += episode.transitions
        elif isinstance(dataset[0], Transition):
            transitions = list(cast(List[Transition], dataset))
        else:
            raise ValueError(f"invalid dataset type: {type(dataset)}")

        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif transitions[0].is_discrete:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

        iterator: TransitionIterator
        print(f'N_EPOCHS: {n_epochs}, N_STEPS: {n_steps}')
        if n_epochs is None and n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
            LOG.debug("RandomIterator is selected.")
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
            LOG.debug("RoundIterator is selected.")
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger

        # initialize scaler
        if self._scaler:
            LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
            self._scaler.fit(transitions)

        # initialize action scaler
        if self._action_scaler:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._action_scaler.get_type(),
            )
            self._action_scaler.fit(transitions)

        # initialize reward scaler
        if self._reward_scaler:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._reward_scaler.get_type(),
            )
            self._reward_scaler.fit(transitions)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = transition.get_action_size()
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            iterator.reset()

            for itr in range_gen:

                # generate new transitions with dynamics models
                new_transitions = self.generate_new_data(
                    transitions=iterator.transitions,
                )
                if new_transitions:
                    iterator.add_generated_transitions(new_transitions)
                    LOG.debug(
                        f"{len(new_transitions)} transitions are generated.",
                        real_transitions=len(iterator.transitions),
                        fake_transitions=len(iterator.generated_transitions),
                    )

                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            self._loss_history["epoch"].append(epoch)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger.close()
        self._active_logger = None

class TBCQM(AlgoBase):
    '''
    TBCQ is the very first practical data-driven deep reinforcement learning
    '''
    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _rl_start_step: int
    _beta: float
    _use_gpu: Optional[Device]
    _impl: Optional[TBCQMImpl]

    _k: int
    _tl: int
    _net_type: str
    _num_layers: int

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        imitator_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        lam: float = 0.75,
        n_action_samples: int = 20,
        action_flexibility: float = 0.05,
        rl_start_step: int = 0,
        beta: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[TBCQMImpl] = None,
        k: int = 128,
        tl: int = 20,
        net_type: str = "GRU",
        num_layers: int = 1,
        **kwargs: Any
    ):
        print('Implimentation of TBCQ!')
        print(f'batch_size: {batch_size}')
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._imitator_learning_rate = imitator_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_actor_interval = update_actor_interval
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._rl_start_step = rl_start_step
        self._beta = beta
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._k = k
        self._tl = tl
        self._net_type = net_type
        self._num_layers = num_layers

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = TBCQMImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            action_flexibility=self._action_flexibility,
            beta=self._beta,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            k=self._k,
            tl=self._tl,
            net_type=self._net_type,
            num_layers=self._num_layers
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # XXX: debug
        # print(f'batch n_steps: {batch.n_steps}')
        # __import__('pprint').pprint(f'batch observations: {batch.observations.shape}')
        # print(f'observation shape: {self.observation_shape}')

        imitator_loss = self._impl.update_imitator(batch)
        metrics.update({"imitator_loss": imitator_loss})

        if self._grad_step >= self._rl_start_step:
            critic_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})

            if self._grad_step % self._update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch)
                metrics.update({"actor_loss": actor_loss})
                self._impl.update_actor_target()
                self._impl.update_critic_target()

        return metrics

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """BCQ does not support sampling action."""
        raise NotImplementedError("BCQ does not support sampling action.")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def _evaluate(self, episodes: List[Episode], scorers: Dict[str, Callable[[Any, List[Episode]], float]], logger: D3RLPyLogger) -> None:
        # __import__('pprint').pprint(scorers)
        return super()._evaluate(episodes, scorers, logger)
