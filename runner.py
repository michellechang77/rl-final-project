from trainer import train, evaluate, preview_env
import os
from stable_baselines3 import PPO

# %%
# INITIALIZE GLObALS
mode = "fast"
# %%
if mode == "fast":
    episodes = 1000
    train_batch_size = 100
    n_steps = 2048
    num_iterations = 10
    total_timesteps = 1000
else:
    episodes = 1000
    train_batch_size = 4000
    n_steps = 4096
    num_iterations = 75
    train_batch_size = 4096
    total_timesteps = 300_000
ppo_args = {"n_steps": n_steps, "batch_size": train_batch_size, "verbose": 1}
learn_args = {"total_timesteps": total_timesteps, "log_interval": 4}
# %%
# intiial training ########################
model = None
meta_drive_env_dict = None
trial_name = "init"

preview_env(meta_drive_env_dict)

model = train(model, meta_drive_env_dict)
gif, total_reward = evaluate(model, episodes, meta_drive_env_dict, trial_name)
# display(gif)
print(total_reward)
model.save(os.path.join(trial_name, "model"))
# %%
del model

# %%
# retrain on new environment ##############################
model = PPO.load(os.path.join(trial_name, "model"))
meta_drive_env_dict = dict(
    map="OC",
    # This policy setting simplifies the task
    discrete_action=True,
    discrete_throttle_dim=3,
    discrete_steering_dim=3,
    horizon=500,
    # scenario setting
    random_spawn_lane_index=False,
    num_scenarios=1,
    start_seed=5,
    traffic_density=0.1,
    accident_prob=0,
    log_level=50,
)
trial_name2 = "init2"

preview_env(meta_drive_env_dict)

model = train(model, meta_drive_env_dict)
gif, total_reward = evaluate(model, episodes, meta_drive_env_dict, trial_name2)


# TODO: set the environment sequences we care about
