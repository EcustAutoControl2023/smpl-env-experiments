import os
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

# set up the default recipe
recipe_dict = {
    FS: Recipe(FS_DEFAULT_PROFILE, FS), # 糖进给率
    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL), # 大豆油进给率
    FG: Recipe(FG_DEFAULT_PROFILE, FG), # 空气的体积流量
    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES), # 压力
    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
}

recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
# set up the environment
normalize = False
env = PenSimEnvGym(recipe_combo=recipe_combo, normalize=normalize)
state = env.reset(normalize=normalize, random_seed_ref=6886)
# load one batch of the sampled data
load_just_a_file = "../extern-lib/smpl/smpl/configdata/pensimenv/random_batch_0.csv"
dataset_obj = PeniControlData(load_just_a_file=load_just_a_file, normalize=normalize)
if dataset_obj.file_list:
    print("Penicillin_Control_Challenge data correctly initialized.")
else:
    raise ValueError("Penicillin_Control_Challenge data initialization failed.")
dataset = dataset_obj.get_dataset()

total_reward = 0.0
for step in range(NUM_STEPS):
    state, reward, done, info = env.step(dataset["actions"][step].tolist())
    total_reward += reward
    if step % 1000 == 0:
        print("reward, total_reward:", reward, total_reward)
print("your total reward is (by default, should be around 3224):", total_reward)
