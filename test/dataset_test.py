import numpy as np
from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy_patch.dataset import TMDPDataset, TEpisode

time_length = 20

# 1000 steps of observations with shape of (100,)
observations = np.random.random((128, time_length, 9))
# 1000 steps of actions with shape of (4,)
actions = np.random.random((128, time_length, 6))
# 1000 steps of rewards
rewards = np.random.random((128, time_length))
# 1000 steps of terminal flags
terminals = np.random.randint(2, size=(128, time_length))


dataset = TMDPDataset(observations, actions, rewards, terminals, time_length)

print(dataset.get_observation_shape())
print(dataset.get_action_size())
print(dataset.get_time_length())

print(isinstance(dataset, MDPDataset))
print(isinstance(dataset.episodes[0], Episode))
print(isinstance(dataset.episodes[0], TEpisode))

print(dataset.episodes[0].get_time_length())

# automatically splitted into d3rlpy.dataset.Episode objects
dataset.episodes

# each episode is also splitted into d3rlpy.dataset.Transition objects
episode = dataset.episodes[0]
episode[0].observation
episode[0].action
episode[0].reward
episode[0].next_observation
episode[0].terminal

# d3rlpy.dataset.Transition object has pointers to previous and next
# transitions like linked list.
# transition = episode[0]
# while transition.next_transition:
#     transition = transition.next_transition
