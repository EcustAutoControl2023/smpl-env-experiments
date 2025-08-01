import os
from d3rlpy.types import Observation
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

# from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData, NUM_STEPS
from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData, NUM_STEPS

# import an expert
from d3rlpy_patch.algos import StaticRecipeExpert, SACIFConfig
import d3rlpy

# set up the default recipe
recipe_dict = {
    FS: Recipe(FS_DEFAULT_PROFILE, FS),  # 糖进给率
    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),  # 大豆油进给率
    FG: Recipe(FG_DEFAULT_PROFILE, FG),  # 空气的体积流量
    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),  # 压力
    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
}

recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
# set up the environment
normalize = False
env = PenSimEnvGym(recipe_combo=recipe_combo, normalize=normalize, dense_reward=True)
# load one batch of the sampled data
load_just_a_file = "./extern-lib/smpl/smpl/configdata/pensimenv/random_batch_0.csv"
dataset_obj = PeniControlData(load_just_a_file=load_just_a_file, normalize=normalize)
if dataset_obj.file_list:
    print("Penicillin_Control_Challenge data correctly initialized.")
else:
    raise ValueError("Penicillin_Control_Challenge data initialization failed.")
dataset = dataset_obj.get_dataset()

# create an expert
expert = StaticRecipeExpert(recipe_dict=dataset)
cql = d3rlpy.load_learnable("./test/d3rlpy_logs/CQL_Pensim_4042_20250315102559/model_10350.d3")
sacif = SACIFConfig(
    batch_size=32,
    observation_scaler=cql.observation_scaler,
    action_scaler=cql.action_scaler,
    reward_scaler=cql.reward_scaler,
).create(device=None)
sacif.build_with_env(env)

total_reward = 0.0
print(NUM_STEPS)

for i in range(1):
    total_reward = 0.0
    state = env.reset(normalize=normalize, random_seed_ref=i)
    action_list = []
    for step in range(NUM_STEPS):
        # raw_action = [value for value in recipe_combo.get_values_dict_at(step).values()]
        # action = [
        #     raw_action[1],
        #     raw_action[2],
        #     raw_action[3],
        #     raw_action[4],
        #     raw_action[0],
        #     raw_action[5],
        # ]
        # state, reward, done, done, info = env.step(action)
        state, reward, done, done, info = env.step(dataset["actions"][step].tolist())
        # state, reward, done, done, info = env.step(expert.guide(algo=sacif, x=state, step=step))
        total_reward += reward

        action_list.append(dataset["actions"][step].tolist())
        # if step % 1000 == 0:
        #     print("reward, total_reward:", reward, total_reward)
        # if step >= 1000:
        #     print(f"step: {step}, obs:{state}")
    print("your total reward is (by default, should be around 3224):", total_reward)
#     a_name_list = [ "Discharge rate",
#                 "Sugar feed rate",
#                 "Soil bean feed rate",
#                 "Aeration rate",
#                 "Back pressure",
#                 "Water injection dilution",
#     ]  # discharge, Fs, Foil, Fg, pressure, Fw
#     for n_a in range(6):
#         a_name = a_name_list[n_a]

#         plt.close("all")
#         plt.figure(0)
#         plt.title(f"{a_name}")
#         alpha = 1
#         algo_name = "algo"
#         plt.plot(
#             np.array(action_list)[:, n_a],
#             label=algo_name,
#             alpha=alpha,
#         )
#         plt.xticks(np.arange(1, 1150 + 2, 1))
#         plt.legend()
#         plot_dir = "./action_plt_dir/"
#         if plot_dir is not None:
#             path_name = os.path.join(
#                 plot_dir, f"baseline_action_{a_name}.png"
#             )
#             plt.savefig(path_name)
#         plt.close()
# # print("your total reward is (by default, should be around 3224):", total_reward/10)
