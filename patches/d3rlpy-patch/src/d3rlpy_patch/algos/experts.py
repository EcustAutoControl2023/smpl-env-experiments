from typing import Any, Dict
from abc import ABCMeta, abstractmethod
import numpy as np

from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.types import NDArray, Observation


class Expert(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def guide(self, algo: QLearningAlgoProtocol, x: Observation, step: int) -> NDArray:
        pass


class StaticRecipeExpert(Expert):
    def __init__(
        self,
        recipe_dict: Dict[str, Any],
    ):
        super().__init__()
        self.recipe_dict = recipe_dict

    def get_values_dict_at(self, time: float) -> Dict:
        """
        Get value of each recipe at given time
        """
        values_dict = {}
        for name, recipe in self.recipe_dict.items():
            values_dict[name] = recipe.get_value_at(time=time)

        return values_dict

    def guide(self, algo: QLearningAlgoProtocol, x: Observation, step: int) -> NDArray:
        # first we get the action size
        action_size = algo.action_size
        assert action_size is not None
        # then we get a dictionary of values for each recipe at the current time
        return np.array(self.recipe_dict["actions"][step].tolist())
