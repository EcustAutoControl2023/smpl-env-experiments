import d3rlpy
from d3rlpy.algos import SACConfig
from d3rlpy.preprocessing import (
    MinMaxActionScaler,
    StandardObservationScaler,
    StandardRewardScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
)
from d3rlpy_patch.algos import SACIFConfig, SACIF
from d3rlpy_patch.algos.experts import Expert, StaticRecipeExpert
import gym
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
import pickle

from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData, NUM_STEPS


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
        )
    else:
        raise ValueError("env_name not recognized")
    return env


def get_pensim_dataset():
    env_name = "pensimenv"
    training_dataset_loc = (
        "./offline_temporal_datasets/pensimenv/900_normalize=False.pkl"
    )
    eval_dataset_loc = "./offline_temporal_datasets/pensimenv/110_normalize=False.pkl"
    seed = 0

    env_config = {
        "env_name": env_name,
        "normalize": False,
        "dense_reward": True,
    }

    env = env_creator(env_config)
    eval_env = env_creator(env_config)
    env.reset()
    d3rlpy.seed(seed)

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

    return env, env_name, seed, dataset, eval_dataset, eval_env


env, env_name, seed, dataset, eval_dataset, eval_env = get_pensim_dataset()
# cql = d3rlpy.algos.CQLConfig(
#     # bc = TBCQConfig(
#     observation_scaler=StandardObservationScaler(),
#     action_scaler=MinMaxActionScaler(),
#     reward_scaler=MinMaxRewardScaler(),
# ).create(device=None)
# print(len(dataset.buffer.episodes) * len(dataset.buffer.episodes[0]))
# # assert False
# cql.fit(
#     dataset,
#     n_steps=135 * 1,
#     n_steps_per_epoch=135,
#     save_interval=1,
#     evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)},
#     experiment_name=f"CQL_Pensim_{seed}",
# )

cql = d3rlpy.load_learnable("./d3rlpy_logs/CQL_Pensim_0_20250307102648/model_135.d3")
# cql = d3rlpy.algos.CQL.from_json(
#     "./d3rlpy_logs/CQL_Pensim_0_20250306152929/params.json"
# )

# sacif = SACIFConfig(
#     batch_size=32,
#     observation_scaler=StandardObservationScaler(),
#     action_scaler=MinMaxActionScaler(),
#     reward_scaler=MinMaxRewardScaler(),
# ).create(device=None)
sacif = SACIFConfig(
    batch_size=32,
    observation_scaler=cql.observation_scaler,
    action_scaler=cql.action_scaler,
    reward_scaler=cql.reward_scaler,
).create(device=None)

# sacif.build_with_env(env)

buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)

recipe_dict = {
    FS: Recipe(FS_DEFAULT_PROFILE, FS),  # 糖进给率
    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),  # 大豆油进给率
    FG: Recipe(FG_DEFAULT_PROFILE, FG),  # 空气的体积流量
    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),  # 压力
    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
}

load_just_a_file = "../extern-lib/smpl/smpl/configdata/pensimenv/random_batch_0.csv"
dataset_obj = PeniControlData(load_just_a_file=load_just_a_file, normalize=False)
if dataset_obj.file_list:
    print("Penicillin_Control_Challenge data correctly initialized.")
else:
    raise ValueError("Penicillin_Control_Challenge data initialization failed.")
recipe_dict = dataset_obj.get_dataset()


sacif.build_with_dataset(dataset)
sacif.copy_policy_from(cql)
sacif.copy_q_function_from(cql)

# sacif.fit(
#     dataset,
#     n_steps=1000,
#     n_steps_per_epoch=100,
#     save_interval=1,
#     evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)},
#     experiment_name=f"SACIF_Pensim_{seed}",
# )

sacif.fit_online(
    env,
    buffer,
    # explorer,
    n_steps=10 * 10,
    eval_env=eval_env,
    n_steps_per_epoch=10,
    update_start_step=10 // 10,
    expert=StaticRecipeExpert(recipe_dict=recipe_dict),
)
