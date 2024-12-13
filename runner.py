# %%
meta_drive_env_dict = None
ppo_args = None
learn_args = None
parallel_envs = 4
model = None
render_args = None
# %%
import trainer
import os
from stable_baselines3 import PPO
import importlib
from IPython.display import display

importlib.reload(trainer)
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    # %%
    # INITIALIZE GLObALS
    mode = "reg"
    # %%
    if mode == "fast":
        episodes = 1000
        n_steps = 4096
        num_iterations = 20
        train_batch_size = 100
        total_timesteps = 50000
    else:
        episodes = 1000
        n_steps = 4096
        num_iterations = 30
        train_batch_size = 4096
        total_timesteps = 300_000
    ppo_args = {
        "n_steps": n_steps,
        "batch_size": train_batch_size,
        "verbose": 1,
        # "policy_kwargs": dict(
        #     net_arch=[128, 128, 128, 128]  # Four layers with 128 units each
        # ),
    }
    learn_args = {"total_timesteps": total_timesteps, "log_interval": 4}
    # meta_drive_env_dict = {
    #     "map": 3,
    #     # This policy setting simplifies the task
    #     "discrete_action": True,
    #     "discrete_throttle_dim": 3,
    #     "discrete_steering_dim": 3,
    #     "horizon": 500,
    #     # scenario setting
    #     "random_spawn_lane_index": False,
    #     "num_scenarios": 1,
    #     "start_seed": 5,
    #     "traffic_density": 0,
    #     "accident_prob": 0,
    #     "log_level": 50,
    # }
    # %%
    # intiial training ########################
    model, gif, total_reward = trainer.execute_trial(
        None,
        meta_drive_env_dict,
        "init4",
        episodes,
        ppo_args,
        learn_args,
        parallel_envs=parallel_envs,
    )
    for frame in range(gif.n_frames):
        gif.seek(frame)
        gif.show()  # This opens each frame in an external viewer
    # # %%
    # model, gif, total_reward = trainer.execute_trial(
    #     None, meta_drive_env_dict, "init", episodes, ppo_args, learn_args
    # )
    # display(gif)
    # # %%
    # difficulty = "easy"
    # dict(
    #     num_scenarios=10,
    #     traffic_density=difficulty,
    #     accident_prob=difficulty,
    #     map_config=map_config,
    #     # random_dynamics=random_dynamics,
    #     log_level=logging.WARNING,
    # )

    # # %%
    # model = None
    # meta_drive_env_dict = None
    # trial_name = "init"

    # trainer.preview_env(meta_drive_env_dict)
    # # %%
    # model = trainer.train(model, meta_drive_env_dict, parallel_envs=1)
    # # %%
    # from stable_baselines3 import PPO

    # model = PPO.load(os.path.join(trial_name, "model.zip"))

    # env = trainer.create_env(meta_drive_env_dict)
    # # Load the model
    # # %%
    # # %%
    # gif, total_reward = trainer.evaluate(
    #     model,
    #     episodes=episodes,
    #     meta_drive_env_dict=meta_drive_env_dict,
    #     trial_name=trial_name,
    # )
    # print(total_reward)
    # model.save(os.path.join(trial_name, "model"))
    # # %%
    # display(gif)
    # del model

    # # %%
    # # retrain on new environment ##############################
    # model = PPO.load(os.path.join(trial_name, "model"))
    # meta_drive_env_dict = dict(
    #     map="OC",
    #     # This policy setting simplifies the task
    #     discrete_action=True,
    #     discrete_throttle_dim=3,
    #     discrete_steering_dim=3,
    #     horizon=500,
    #     # scenario setting
    #     random_spawn_lane_index=False,
    #     num_scenarios=1,
    #     start_seed=5,
    #     traffic_density=0.1,
    #     accident_prob=0,
    #     log_level=50,
    # )
    # trial_name2 = "init2"

    # trainer.preview_env(meta_drive_env_dict)

    # model = trainer.train(model, meta_drive_env_dict)
    # gif, total_reward = trainer.evaluate(model, episodes, meta_drive_env_dict, trial_name2)

    # # TODO: set the environment sequences we care about
