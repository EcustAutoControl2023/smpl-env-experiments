import copy
import warnings

import numpy as np

cimport numpy as np

import cython
import h5py

from cython cimport view

from cython.parallel import prange

from dataset cimport CTransition
from libc.string cimport memcpy, memset
from libcpp cimport bool, nullptr
from libcpp.memory cimport make_shared, shared_ptr

from .logger import LOG

from d3rlpy.dataset import *

# (o_t, a_t, r_t, t_t)
# r_t = r(o_t, a_t)
# observations = [o_t, o_t+1, o_t+2, o_t, o_t+1]
# actions      = [a_t, a_t+1, a_t+2, a_t, a_t+1]
# rewards      = [r_t, r_t+1, r_t+2, r_t, r_t+1]
# terminals    = [  0,     0,     1,   0,     1]

# obs          = [o_t  , o_t+1, o_t  ]
# next_obs     = [o_t+1, o_t+2, o_t+1]
# action       = [a_t  , a_t+1, a_t  ]
# rewards      = [r_t  , r_t+1, r_t  ]
# terminals    = [0    , 1    , 1    ]

def _safe_size(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    elif isinstance(array, np.ndarray):
        return array.shape[0]
    raise ValueError


def _to_episodes(
    observation_shape,
    action_size,
    observations,
    actions,
    rewards,
    terminals,
    time_length,
    episode_terminals,
):
    rets = []
    head_index = 0
    for i in range(_safe_size(observations)):
        if episode_terminals[i]:
            episode = TEpisode(
                observation_shape=observation_shape,
                action_size=action_size,
                observations=observations[head_index:i + 1],
                actions=actions[head_index:i + 1],
                rewards=rewards[head_index:i + 1],
                time_length=time_length,
                terminal=terminals[i],
            )
            rets.append(episode)
            head_index = i + 1
    return rets


def _to_transitions(
    observation_shape,
    action_size,
    observations,
    actions,
    rewards,
    terminal,
):
    rets = []
    num_data = _safe_size(observations)
    prev_transition = None
    for i in range(num_data):
        observation = observations[i]
        action = actions[i]
        reward = rewards[i]

        # XXX: debug
        # print('observation', observation)
        # print('action', action)
        # print('reward', reward)

        if i == num_data - 1:
            if terminal:
                # dummy observation
                next_observation = np.zeros_like(observation)
            else:
                # skip the last step if not terminated
                break
        else:
            next_observation = observations[i + 1]

        env_terminal = terminal if i == num_data - 1 else 0.0

        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=env_terminal,
            prev_transition=prev_transition
        )

        # set pointer to the next transition
        if prev_transition:
            prev_transition.next_transition = transition

        prev_transition = transition

        rets.append(transition)
    return rets


def _check_discrete_action(actions):
    float_actions = np.array(actions, dtype=np.float32)
    int_actions = np.array(actions, dtype=np.int32)
    return np.all(float_actions == int_actions)


class TMDPDataset(MDPDataset):
    # FIXME:
    """Temporal Markov-Decision Process Dataset class.

    MDPDataset is deisnged for reinforcement learning datasets to use them like
    supervised learning datasets.

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset

        # 1000 steps of observations with shape of (100,)
        observations = np.random.random((1000, 100))
        # 1000 steps of actions with shape of (4,)
        actions = np.random.random((1000, 4))
        # 1000 steps of rewards
        rewards = np.random.random(1000)
        # 1000 steps of terminal flags
        terminals = np.random.randint(2, size=1000)

        dataset = MDPDataset(observations, actions, rewards, terminals)

    The MDPDataset object automatically splits the given data into list of
    :class:`d3rlpy.dataset.Episode` objects.
    Furthermore, the MDPDataset object behaves like a list in order to use with
    scikit-learn utilities.

    .. code-block:: python

        # returns the number of episodes
        len(dataset)

        # access to the first episode
        episode = dataset[0]

        # iterate through all episodes
        for episode in dataset:
            pass

    Args:
        observations (numpy.ndarray): N-D array. If the
            observation is a vector, the shape should be
            `(N, dim_observation)`. If the observations is an image, the shape
            should be `(N, C, H, W)`.
        actions (numpy.ndarray): N-D array. If the actions-space is
            continuous, the shape should be `(N, dim_action)`. If the
            action-space is discrete, the shape should be `(N,)`.
        rewards (numpy.ndarray): array of scalar rewards. The reward function
            should be defined as :math:`r_t = r(s_t, a_t)`.
        terminals (numpy.ndarray): array of binary terminal flags.
        time_length (int): time length of temporal data.
        episode_terminals (numpy.ndarray): array of binary episode terminal
            flags. The given data will be splitted based on this flag.
            This is useful if you want to specify the non-environment
            terminations (e.g. timeout). If ``None``, the episode terminations
            match the environment terminations.
        discrete_action (bool): flag to use the given actions as discrete
            action-space actions. If ``None``, the action type is automatically
            determined.

    """
    def __init__(
        self,
        observations,
        actions,
        rewards,
        terminals,
        time_length=1,
        episode_terminals=None,
        discrete_action=None,
    ):
        print('experimental temporal dataset feature for d3rlpy v1.1.1!')
        self._time_length = time_length
        assert isinstance(observations, np.ndarray),\
            'Observations must be numpy array.'
        if observations.dtype != np.float32:
            observations = np.asarray(observations, dtype=np.float32)

        # check nan
        assert np.all(np.logical_not(np.isnan(observations)))
        assert np.all(np.logical_not(np.isnan(actions)))
        assert np.all(np.logical_not(np.isnan(rewards)))
        assert np.all(np.logical_not(np.isnan(terminals)))

        self._observations = observations.reshape(-1, observations.shape[-1])
        self._rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self._terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)

        if episode_terminals is None:
            # if None, episode terminals match the environment terminals
            self._episode_terminals = self._terminals
        else:
            self._episode_terminals = np.asarray(
                episode_terminals, dtype=np.float32)

        # automatic action type detection
        if discrete_action is None:
            discrete_action = _check_discrete_action(actions)

        self.discrete_action = discrete_action
        if discrete_action:
            self._actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            self._actions = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])

        self._episodes = None
        # super().__init__(
        #     observations=observations,
        #     actions=actions,
        #     rewards=rewards,
        #     terminals=terminals,
        #     episode_terminals=episode_terminals,
        #     discrete_action=discrete_action,
        # )

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards

        """
        return self._rewards

    @property
    def terminals(self):
        """ Returns the terminal flags.

        Returns:
            numpy.ndarray: array of terminal flags.

        """
        return self._terminals

    @property
    def episode_terminals(self):
        """ Returns the episode terminal flags.

        Returns:
            numpy.ndarray: array of episode terminal flags.

        """
        return self._episode_terminals

    @property
    def episodes(self):
        """ Returns the episodes.

        Returns:
            list(d3rlpy.dataset.Episode):
                list of :class:`d3rlpy.dataset.Episode` objects.

        """
        if self._episodes is None:
            self.build_episodes()
        return self._episodes

    @property
    def time_length(self):
        """ Returns time length.

        Returns:
            int: time length.

        """
        return self._time_length

    def size(self):
        """ Returns the number of episodes in the dataset.

        Returns:
            int: the number of episodes.

        """
        return len(self.episodes)

    def get_action_size(self):
        """ Returns dimension of action-space.

        If `discrete_action=True`, the return value will be the maximum index
        +1 in the give actions.

        Returns:
            int: dimension of action-space.

        """
        if self.discrete_action:
            return int(np.max(self._actions) + 1)
        return self._actions.shape[-1]

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self._observations[0].shape

    def get_time_length(self):
        """ Returns time length.

        Returns:
            int: time length.

        """
        return self._time_length

    def is_action_discrete(self):
        """ Returns `discrete_action` flag.

        Returns:
            bool: `discrete_action` flag.

        """
        return self.discrete_action

    def compute_stats(self):
        """ Computes statistics of the dataset.

        .. code-block:: python

            stats = dataset.compute_stats()

            # return statistics
            stats['return']['mean']
            stats['return']['std']
            stats['return']['min']
            stats['return']['max']

            # reward statistics
            stats['reward']['mean']
            stats['reward']['std']
            stats['reward']['min']
            stats['reward']['max']

            # action (only with continuous control actions)
            stats['action']['mean']
            stats['action']['std']
            stats['action']['min']
            stats['action']['max']

            # observation (only with numpy.ndarray observations)
            stats['observation']['mean']
            stats['observation']['std']
            stats['observation']['min']
            stats['observation']['max']

        Returns:
            dict: statistics of the dataset.

        """
        episode_returns = []
        for episode in self.episodes:
            episode_returns.append(episode.compute_return())

        stats = {
            'return': {
                'mean': np.mean(episode_returns),
                'std': np.std(episode_returns),
                'min': np.min(episode_returns),
                'max': np.max(episode_returns),
                'histogram': np.histogram(episode_returns, bins=20)
            },
            'reward': {
                'mean': np.mean(self._rewards),
                'std': np.std(self._rewards),
                'min': np.min(self._rewards),
                'max': np.max(self._rewards),
                'histogram': np.histogram(self._rewards, bins=20)
            }
        }

        # only for continuous control task
        if not self.discrete_action:
            # calculate histogram on each dimension
            hists = []
            for i in range(self.get_action_size()):
                hists.append(np.histogram(self.actions[:, i], bins=20))
            stats['action'] = {
                'mean': np.mean(self.actions, axis=0),
                'std': np.std(self.actions, axis=0),
                'min': np.min(self.actions, axis=0),
                'max': np.max(self.actions, axis=0),
                'histogram': hists
            }
        else:
            # count frequency of discrete actions
            freqs = []
            for i in range(self.get_action_size()):
                freqs.append((self.actions == i).sum())
            stats['action'] = {
                'histogram': [freqs, np.arange(self.get_action_size())]
            }

        # avoid large copy when observations are huge data.
        stats['observation'] = {
            'mean': np.mean(self.observations, axis=0),
            'std': np.std(self.observations, axis=0),
            'min': np.min(self.observations, axis=0),
            'max': np.max(self.observations, axis=0),
        }

        return stats

    def append(
        self,
        observations,
        actions,
        rewards,
        terminals,
        episode_terminals=None
    ):
        """ Appends new data.

        Args:
            observations (numpy.ndarray): N-D array.
            actions (numpy.ndarray): actions.
            rewards (numpy.ndarray): rewards.
            terminals (numpy.ndarray): terminals.
            episode_terminals (numpy.ndarray): episode terminals.

        """
        # validation
        for observation, action in zip(observations, actions):
            assert observation.shape == self.get_observation_shape(),\
                f'Observation shape must be {self.get_observation_shape()}.'
            if self.discrete_action:
                if int(action) >= self.get_action_size():
                    message = f'New action size is higher than' \
                              f' {self.get_action_size()}.'
                    warnings.warn(message)
            else:
                assert action.shape == (self.get_action_size(), ),\
                    f'Action size must be {self.get_action_size()}.'

        # append observations
        self._observations = np.vstack([self._observations, observations])

        # append actions
        if self.discrete_action:
            self._actions = np.hstack([self._actions, actions])
        else:
            self._actions = np.vstack([self._actions, actions])

        # append rests
        self._rewards = np.hstack([self._rewards, rewards])
        self._terminals = np.hstack([self._terminals, terminals])
        if episode_terminals is None:
            episode_terminals = terminals
        self._episode_terminals = np.hstack(
            [self._episode_terminals, episode_terminals]
        )


        # convert new data to list of episodes
        episodes = _to_episodes(
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            time_length=self._time_length,
            episode_terminals=self._episode_terminals,
        )

        self._episodes = episodes

    def extend(self, dataset):
        """ Extend dataset by another dataset.

        Args:
            dataset (d3rlpy.dataset.MDPDataset): dataset.

        """
        assert self.is_action_discrete() == dataset.is_action_discrete(),\
            'Dataset must have discrete action-space.'
        assert self.get_observation_shape() == dataset.get_observation_shape(),\
            f'Observation shape must be {self.get_observation_shape()}'

        self.append(
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.terminals,
            dataset.episode_terminals
        )

    def dump(self, fname):
        """ Saves dataset as HDF5.

        Args:
            fname (str): file path.

        """
        with h5py.File(fname, 'w') as f:
            f.create_dataset('observations', data=self._observations)
            f.create_dataset('actions', data=self._actions)
            f.create_dataset('rewards', data=self._rewards)
            f.create_dataset('terminals', data=self._terminals)
            f.create_dataset('episode_terminals', data=self._episode_terminals)
            f.create_dataset('discrete_action', data=self.discrete_action)
            f.create_dataset('version', data='1.0')
            f.flush()

    @classmethod
    def load(cls, fname):
        """ Loads dataset from HDF5.

        .. code-block:: python

            import numpy as np
            from d3rlpy.dataset import MDPDataset

            dataset = MDPDataset(np.random.random(10, 4),
                                 np.random.random(10, 2),
                                 np.random.random(10),
                                 np.random.randint(2, size=10))

            # save as HDF5
            dataset.dump('dataset.h5')

            # load from HDF5
            new_dataset = MDPDataset.load('dataset.h5')

        Args:
            fname (str): file path.

        """
        with h5py.File(fname, 'r') as f:
            observations = f['observations'][()]
            actions = f['actions'][()]
            rewards = f['rewards'][()]
            terminals = f['terminals'][()]
            discrete_action = f['discrete_action'][()]

            # for backward compatibility
            if 'episode_terminals' in f:
                episode_terminals = f['episode_terminals'][()]
            else:
                episode_terminals = None

            if 'version' not in f:
                LOG.warning("The dataset structure might be incompatible.")

        dataset = cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
            discrete_action=discrete_action,
        )

        return dataset

    def build_episodes(self):
        """ Builds episode objects.

        This method will be internally called when accessing the episodes
        property at the first time.

        """
        self._episodes = _to_episodes(
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            time_length=self._time_length,
            episode_terminals=self._episode_terminals,
        )

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.episodes[index]

    def __iter__(self):
        return iter(self.episodes)


class TEpisode(Episode):
    # FIXME:
    """ Temporal Episode class.

    This class is designed to hold data collected in a single episode.

    Episode object automatically splits data into list of
    :class:`d3rlpy.dataset.Transition` objects.
    Also Episode object behaves like a list object for ease of access to
    transitions.

    .. code-block:: python

        # return the number of transitions
        len(episode)

        # access to the first transition
        transitions = episode[0]

        # iterate through all transitions
        for transition in episode:
            pass

    Args:
        observation_shape (tuple): observation shape.
        action_size (int): dimension of action-space.
        observations (numpy.ndarray): observations.
        actions (numpy.ndarray): actions.
        rewards (numpy.ndarray): scalar rewards.
        terminal (bool): binary terminal flag. If False, the episode is not
            terminated by the environment (e.g. timeout).

    """
    def __init__(
        self,
        observation_shape,
        action_size,
        observations,
        actions,
        rewards,
        time_length=1,
        terminal=True,
    ):
        self._time_length = time_length
        assert isinstance(observations, np.ndarray),\
            'Observation must be numpy array.'
        if observations.dtype != np.float32:
            observations = np.asarray(observations, dtype=np.float32)

        # fix action dtype and shape
        if len(actions.shape) == 1:
            actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            actions = np.asarray(actions, dtype=np.float32)

        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observations = observations
        self._actions = actions
        self._rewards = np.asarray(rewards, dtype=np.float32)
        self._terminal = terminal
        self._transitions = None

        # super().__init__(
        #     observation_shape=observation_shape,
        #     action_size=action_size,
        #     observations=observations,
        #     actions=actions,
        #     rewards=rewards,
        #     terminal=terminal,
        # )

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards.

        """
        return self._rewards

    @property
    def terminal(self):
        """ Returns the terminal flag.

        Returns:
            bool: the terminal flag.

        """
        return self._terminal

    @property
    def transitions(self):
        """ Returns the transitions.

        Returns:
            list(d3rlpy.dataset.Transition):
                list of :class:`d3rlpy.dataset.Transition` objects.

        """
        if self._transitions is None:
            self.build_transitions()
        return self._transitions

    @property
    def time_length(self):
        """ Returns time length.

        Returns:
            int: time length.

        """
        return self._time_length

    def build_transitions(self):
        """ Builds transition objects.

        This method will be internally called when accessing the transitions
        property at the first time.

        """
        self._transitions = _to_transitions(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminal=self._terminal,
        )

    def size(self):
        """ Returns the number of transitions.

        Returns:
            int: the number of transitions.

        """
        return len(self.transitions)

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self.observation_shape

    def get_action_size(self):
        """ Returns dimension of action-space.

        Returns:
            int: dimension of action-space.

        """
        return self.action_size

    def get_time_length(self):
        """ Returns time length.

        Returns:
            int: time length.

        """
        return self._time_length

    def compute_return(self):
        """ Computes sum of rewards.

        .. math::

            R = \\sum_{i=1} r_i

        Returns:
            float: episode return.

        """
        return np.sum(self._rewards)

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.transitions[index]

    def __iter__(self):
        return iter(self.transitions)

