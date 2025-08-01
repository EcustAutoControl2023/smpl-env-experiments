import os
import numpy as np
import matplotlib.pyplot as plt

from d3rlpy.interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from d3rlpy.types import GymEnv

__all__ = [
    "evaluate_qlearning_with_environment_and_plot",
    # "evaluate_transformer_with_environment",
]

def evaluate_qlearning_with_environment_and_plot(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
    plot: bool = False,
    plot_dir: str = './plt_dir',
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward = 0.0
        actions_list = []

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(observation, np.ndarray):
                    observation = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation = [
                        np.expand_dims(o, axis=0) for o in observation
                    ]
                else:
                    raise ValueError(
                        f"Unsupported observation type: {type(observation)}"
                    )
                action = algo.predict(observation)[0]
            actions_list.append(action)

            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if done or truncated:
                if plot:
                    # plot observations
                    # for n_o in range(9):
                    #     o_name = self.observation_name[n_o]
                    #
                    #     plt.close("all")
                    #     plt.figure(0)
                    #     plt.title(f"{o_name}")
                    #     for n_algo in range(len(algorithms)):
                    #         alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    #         _, algo_name, _ = algorithms[n_algo]
                    #         plt.plot(
                    #             np.array(observations_list[n_algo][-1])[:, n_o],
                    #             label=algo_name,
                    #             alpha=alpha,
                    #         )
                    #     plt.plot(
                    #         [initial_states[n_epi][n_o] for _ in range(self.max_steps)],
                    #         linestyle="--",
                    #         label=f"initial_{o_name}",
                    #     )
                    #     plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    #     plt.annotate(
                    #         str(initial_states[n_epi][n_o]),
                    #         xy=(0, initial_states[n_epi][n_o]),
                    #     )
                    #     plt.legend()
                    #     if plot_dir is not None:
                    #         path_name = os.path.join(
                    #             plot_dir, f"{n_epi}_observation_{o_name}.png"
                    #         )
                    #         plt.savefig(path_name)
                    #     plt.close()

                    # plot actions
                    a_name_list = [ "Discharge rate",
                                "Sugar feed rate",
                                "Soil bean feed rate",
                                "Aeration rate",
                                "Back pressure",
                                "Water injection dilution",
                    ]  # discharge, Fs, Foil, Fg, pressure, Fw
                    for n_a in range(6):
                        a_name = a_name_list[n_a]

                        plt.close("all")
                        plt.figure(0)
                        plt.title(f"{a_name}")
                        alpha = 1
                        algo_name = "algo"
                        plt.plot(
                            np.array(actions_list)[:, n_a],
                            label=algo_name,
                            alpha=alpha,
                        )
                        plt.xticks(np.arange(1, 1150 + 2, 1))
                        plt.legend()
                        if plot_dir is not None:
                            path_name = os.path.join(
                                plot_dir, f"trial1_action_{a_name}.png"
                            )
                            plt.savefig(path_name)
                        plt.close()

                    # # plot rewards
                    # plt.close("all")
                    # plt.figure(0)
                    # plt.title("reward")
                    # for n_algo in range(len(algorithms)):
                    #     alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    #     _, algo_name, _ = algorithms[n_algo]
                    #     plt.plot(
                    #         np.array(rewards_list[n_algo][-1]), label=algo_name, alpha=alpha
                    #     )
                    # plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    # plt.legend()
                    # if plot_dir is not None:
                    #     path_name = os.path.join(plot_dir, f"{n_epi}_reward.png")
                    #     plt.savefig(path_name)
                    # plt.close()
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))
