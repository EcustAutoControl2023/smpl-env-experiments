from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import d3rlpy
import pickle
from smpl.envs.pensimenv import PeniControlData
from pensimpy.examples.recipe import Recipe
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import (
    FS_DEFAULT_PROFILE,
    FOIL_DEFAULT_PROFILE,
    FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE,
    DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE,
    PAA_DEFAULT_PROFILE,
)


def set_env_config(
    env_name,
    normalize=None,
    dense_reward=None,
    reward_on_steady=None,
    reward_on_absolute_efactor=None,
    compute_diffs_on_reward=None,
    standard_reward_style=None,
    initial_state_deviation_ratio=None,
    seed=None,
):
    if env_name == "pensimenv":
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
            "random_seed": seed,
        }
    elif env_name == "beerfmtenv":
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    elif env_name == "atropineenv":
        assert normalize is not None
        assert dense_reward is not None
        assert reward_on_steady is not None
        assert reward_on_absolute_efactor is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "reward_on_steady": False,
            "reward_on_absolute_efactor": True,
        }
    elif env_name == "reactorenv":
        assert normalize is not None
        assert dense_reward is not None
        assert compute_diffs_on_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
            "compute_diffs_on_reward": compute_diffs_on_reward,
            "random_seed": seed,
        }
    elif env_name == "mabupstreamenv":
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    elif env_name == "mabenv":
        assert normalize is not None
        assert dense_reward is not None
        assert standard_reward_style is not None
        assert initial_state_deviation_ratio is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
            "standard_reward_style": standard_reward_style,
            "initial_state_deviation_ratio": initial_state_deviation_ratio,
        }
    elif env_name == "smbenv":
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    else:
        raise ValueError("env_name not recognized")
    return env_config


def env_creator(env_config):
    """
    so that all environments are created in the same way, in training and inference.
    has to be in online_experiments, otherwise will trigger ModuleNotFoundError: No module named 'models'
    in ray/serialization.py
    """

    if env_config["env_name"] == "pensimenv":
        from pensimpy.examples.recipe import Recipe, RecipeCombo
        from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
        from pensimpy.data.constants import (
            FS_DEFAULT_PROFILE,
            FOIL_DEFAULT_PROFILE,
            FG_DEFAULT_PROFILE,
            PRESS_DEFAULT_PROFILE,
            DISCHARGE_DEFAULT_PROFILE,
            WATER_DEFAULT_PROFILE,
            PAA_DEFAULT_PROFILE,
        )
        from smpl.envs.pensimenv import PenSimEnvGym

        recipe_dict = {
            FS: Recipe(FS_DEFAULT_PROFILE, FS),
            FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
            FG: Recipe(FG_DEFAULT_PROFILE, FG),
            PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
            DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
            WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
            PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
        }
        recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
        # set up the environment
        env = PenSimEnvGym(
            recipe_combo=recipe_combo,
            normalize=env_config["normalize"],
            dense_reward=env_config["dense_reward"],
            random_seed=env_config["random_seed"],
        )
    elif env_config["env_name"] == "mabenv":
        from smpl.envs.mabenv import MAbEnvGym

        env = MAbEnvGym(
            normalize=env_config["normalize"],
            dense_reward=env_config["dense_reward"],
            standard_reward_style=env_config["standard_reward_style"],
            initial_state_deviation_ratio=env_config["initial_state_deviation_ratio"],
            random_seed=env_config["random_seed"],
        )
    elif env_config["env_name"] == "atropineenv":
        from smpl.envs.atropineenv import AtropineEnvGym

        env = AtropineEnvGym(
            normalize=env_config["normalize"], reward_scaler=100000
        )  # by default uses reward on steady.
    elif env_config["env_name"] == "reactorenv":
        from smpl.envs.reactorenv import ReactorEnvGym

        env = ReactorEnvGym(
            normalize=env_config["normalize"],
            dense_reward=env_config["dense_reward"],
            compute_diffs_on_reward=env_config["compute_diffs_on_reward"],
            random_seed=env_config["random_seed"],
        )
        # env.reward_function = env_config["reward_function"](env)
    elif env_config["env_name"] == "smbenv":
        from smpl.envs.smbenv import SMBEnvGym

        env = SMBEnvGym(
            normalize=env_config["normalize"],
            dense_reward=env_config["dense_reward"],
        )
    else:
        raise ValueError("env_name not recognized")
    return env


def plot_dataset(dataset, plot: bool = False):
    if plot:
        cumulative_rewards = dataset["rewards"]
        terminal = dataset["terminals"]
        print("Cumulative Rewards:", cumulative_rewards.shape)
        print("Terminal States:", terminal.shape)

        # 绘制奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label="Cumulative Rewards", color="blue")
        plt.plot(terminal, label="Terminal States", color="red", linestyle="--")
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Rewards")
        plt.title("Reward Curve")
        plt.legend()
        plt.grid()
        plt.show()


def get_datasets(env_name: str, dataset_path: Optional[str] = None, plot: bool = False):
    print(f"dataset_path: {dataset_path}")
    training_dataset_loc = None
    # get file real path
    file_path = __file__
    file_dir = file_path.rsplit("/", 1)[0]
    if env_name == "mabenv":
        if dataset_path is None:
            training_dataset_loc = (
                f"{file_dir}/offline_datasets/mabenv/mpc/1000_normalize=False.pkl"
            )
        else:
            training_dataset_loc = dataset_path

        eval_dataset_loc = (
            f"{file_dir}/offline_datasets/mabenv/mpc/100_normalize=False.pkl"
        )

        with open(training_dataset_loc, "rb") as handle:
            training_dataset_pkl = pickle.load(handle)
        with open(eval_dataset_loc, "rb") as handle:
            eval_dataset_pkl = pickle.load(handle)

        dataset = d3rlpy.dataset.MDPDataset(
            training_dataset_pkl["observations"],
            training_dataset_pkl["actions"],
            training_dataset_pkl["rewards"],
            training_dataset_pkl["terminals"],
        )
        eval_dataset = d3rlpy.dataset.MDPDataset(
            eval_dataset_pkl["observations"],
            eval_dataset_pkl["actions"],
            eval_dataset_pkl["rewards"],
            eval_dataset_pkl["terminals"],
        )
        plot_dataset(dataset, plot)
        return dataset
    elif env_name == "smbenv":
        if dataset_path is not None:
            training_dataset_loc = dataset_path
        else:
            training_dataset_loc = (
                f"{file_dir}/offline_datasets/smbenv/mpc/1000_normalize=False.pkl"
            )

        with open(training_dataset_loc, "rb") as handle:
            training_dataset_pkl = pickle.load(handle)

        dataset = d3rlpy.dataset.MDPDataset(
            training_dataset_pkl["observations"],
            training_dataset_pkl["actions"],
            training_dataset_pkl["rewards"],
            training_dataset_pkl["terminals"],
        )
        plot_dataset(dataset, plot)
        return dataset
    elif env_name == "reactorenv":
        if dataset_path is not None:
            training_dataset_loc = dataset_path
        else:
            training_dataset_loc = f"{file_dir}/offline_datasets/reactorenv/mpc_step_50_normalize=False.pkl"
        eval_dataset_loc = (
            f"{file_dir}/offline_datasets/reactorenv/100_normalize=False.pkl"
        )

        with open(training_dataset_loc, "rb") as handle:
            training_dataset_pkl = pickle.load(handle)

        dataset = d3rlpy.dataset.MDPDataset(
            training_dataset_pkl["observations"],
            training_dataset_pkl["actions"],
            training_dataset_pkl["rewards"],
            training_dataset_pkl["terminals"],
        )
        plot_dataset(dataset, plot)
        return dataset
    elif env_name == "atropineenv":
        if dataset_path is not None:
            training_dataset_loc = dataset_path
        else:
            training_dataset_loc = (
                f"{file_dir}/offline_datasets/atropineenv/10000_normalize=False.pkl"
            )
        eval_dataset_loc = (
            f"{file_dir}/offline_datasets/atropineenv/100_normalize=False.pkl"
        )

        with open(training_dataset_loc, "rb") as handle:
            training_dataset_pkl = pickle.load(handle)

        dataset = d3rlpy.dataset.MDPDataset(
            training_dataset_pkl["observations"],
            training_dataset_pkl["actions"],
            training_dataset_pkl["rewards"],
            training_dataset_pkl["terminals"],
        )
        plot_dataset(dataset, plot)
        return dataset
    elif env_name == "pensimenv":
        if dataset_path is not None:
            training_dataset_loc = dataset_path
        else:
            training_dataset_loc = (
                f"{file_dir}/offline_datasets/pensimenv/900_normalize=False.pkl"
            )
        # eval_dataset_loc = (
        #     f"{file_dir}/offline_datasets/pensimenv/110_normalize=False.pkl"
        # )

        with open(training_dataset_loc, "rb") as handle:
            training_dataset_pkl = pickle.load(handle)
        # with open(eval_dataset_loc, "rb") as handle:
        #     eval_dataset_pkl = pickle.load(handle)

        dataset = d3rlpy.dataset.MDPDataset(
            training_dataset_pkl["observations"],
            training_dataset_pkl["actions"],
            training_dataset_pkl["rewards"],
            training_dataset_pkl["terminals"],
        )
        # eval_dataset = d3rlpy.dataset.MDPDataset(
        #     eval_dataset_pkl["observations"],
        #     eval_dataset_pkl["actions"],
        #     eval_dataset_pkl["rewards"],
        #     eval_dataset_pkl["terminals"],
        # )
        if plot:
            cumulative_rewards = eval_dataset_pkl["rewards"]
            terminal = eval_dataset_pkl["terminals"]
            print("Cumulative Rewards:", cumulative_rewards.shape)
            print("Terminal States:", terminal.shape)

            # 绘制奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(cumulative_rewards, label="Cumulative Rewards", color="blue")
            plt.plot(terminal, label="Terminal States", color="red", linestyle="--")
            plt.xlabel("Time Steps")
            plt.ylabel("Cumulative Rewards")
            plt.title("Reward Curve")
            plt.legend()
            plt.grid()
            plt.show()
        return dataset


def get_recipe(env_name: str):
    file_path = __file__
    file_dir = file_path.rsplit("/", 1)[0]
    if env_name == "mabenv":
        return {}
    elif env_name == "pensimenv":
        recipe_dict = {
            FS: Recipe(FS_DEFAULT_PROFILE, FS),  # 糖进给率
            FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),  # 大豆油进给率
            FG: Recipe(FG_DEFAULT_PROFILE, FG),  # 空气的体积流量
            PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),  # 压力
            DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
            WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
            PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
        }

        load_just_a_file = (
            # "../extern-lib/smpl/smpl/configdata/pensimenv/random_batch_0.csv"
            f"{file_dir}/pensimpy_1010_samples/gpei_batch_0.csv"
        )
        dataset_obj = PeniControlData(
            load_just_a_file=load_just_a_file, normalize=False
        )
        if dataset_obj.file_list:
            print("Penicillin_Control_Challenge data correctly initialized.")
        else:
            raise ValueError("Penicillin_Control_Challenge data initialization failed.")
        recipe_dict = dataset_obj.get_dataset()

        return recipe_dict


if __name__ == "__main__":
    env_name = "smbenv"
    dataset = get_datasets(env_name)
    assert dataset is not None, "Dataset is None"
    print(dataset.episodes[0])
