from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from d3rlpy.algos import SAC

@dataclass
class LagrangianSACConfig:
    """Configuration for Lagrangian Actor‑Critic (SAC) algorithm.

    Args:
        cost_threshold: The maximum allowable expected cost.
        lagrangian_lr: Learning rate for updating the Lagrange multiplier.
    """
    cost_threshold: float = 1.0
    lagrangian_lr: float = 0.01

class LagrangianSAC(SAC):
    """Soft Actor‑Critic algorithm with a Lagrangian multiplier to enforce a cost constraint.

    This algorithm augments the reward with a penalty term derived from a cost critic
    and updates a Lagrange multiplier to satisfy an expected cost threshold.  It is a
    simple example to illustrate how safety constraints can be integrated into an
    existing off‑policy actor‑critic method.
    """

    def __init__(
        self,
        cost_threshold: float = 1.0,
        lagrangian_lr: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cost_threshold = cost_threshold
        self.lagrangian_lr = lagrangian_lr
        # Initialize Lagrange multiplier (must be non‑negative)
        self.lambda_param: float = 0.0

    def update(self, batch: Dict[str, np.ndarray]) -> None:
        """Performs a single update of the actor‑critic networks and Lagrangian multiplier.

        The parent SAC update is executed first, then the average cost of the
        batch is used to adjust the Lagrange multiplier.  The cost values must
        be provided in the batch under the key ``'cost'``.

        Args:
            batch: A dictionary containing transition data.  Expected keys are
                those required by the SAC parent class plus an additional
                ``'cost'`` entry with shape (batch_size,).
        """
        # first update the policy and value networks as in SAC
        super().update(batch)

        # update Lagrange multiplier based on cost constraint
        cost = batch.get("cost", None)
        if cost is not None:
            mean_cost = float(np.mean(cost))
            # gradient ascent on the dual function
            self.lambda_param = max(
                0.0,
                self.lambda_param + self.lagrangian_lr * (mean_cost - self.cost_threshold),
            )
