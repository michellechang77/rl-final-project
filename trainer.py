from metadrive.envs import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils import generate_gif
from PIL import Image
from typing import Optional, Dict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from IPython.display import clear_output
import os
from IPython.display import display
import os
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import logging
from tqdm import tqdm


meta_drive_env_dict_default = {
    "map": "C",
    # This policy setting simplifies the task
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 500,
    # scenario setting
    "random_spawn_lane_index": False,
    "num_scenarios": 1,
    "start_seed": 5,
    "traffic_density": 0,
    "accident_prob": 0,
    "log_level": 50,
}

# ppo_args_default =


def preview_env(meta_drive_env_dict: Optional[Dict] = None):
    env = create_env(meta_drive_env_dict=meta_drive_env_dict)
    env.reset()
    ret = env.render(
        mode="topdown", window=False, screen_size=(600, 600), camera_position=(50, 50)
    )
    env.close()
    plt.axis("off")
    plt.imshow(ret)


# Define the environment creation function
def create_env(meta_drive_env_dict: Optional[Dict] = None, need_monitor=False):
    if meta_drive_env_dict is None:
        meta_drive_env_dict = meta_drive_env_dict_default
    env = MetaDriveEnv(meta_drive_env_dict)
    # if need_monitor:
    #     env = Monitor(env)
    return env


def train(
    model: Optional[PPO] = None,
    meta_drive_env_dict: Optional[Dict] = None,
    ppo_args: Optional[Dict] = None,
    learn_args: Optional[Dict] = None,
    parallel_envs=2,
):
    set_random_seed(0)
    # ppo_arg defaults
    if ppo_args is None:
        ppo_args = {"n_steps": 2048, "batch_size": 100, "verbose": 1}
    if learn_args is None:
        learn_args = {"total_timesteps": 1000, "log_interval": 4}
    # train_env = create_env(meta_drive_env_dict)
    # 4 subprocess to rollout
    if parallel_envs > 1:
        train_env = SubprocVecEnv(
            [partial(create_env, meta_drive_env_dict) for _ in range(parallel_envs)]
        )
    else:
        train_env = DummyVecEnv([partial(create_env, meta_drive_env_dict)])
    # If no model is provided, create a new one
    if model is None:
        model = PPO(policy="MlpPolicy", env=train_env, **ppo_args)
    else:
        # Set the new training environment for the existing model
        model.set_env(train_env)
    logging.info("starting learning")
    model.learn(**learn_args)
    train_env.close()
    logging.info("Training is finished!")
    return model


def evaluate(model, meta_drive_env_dict, episodes, trial_name=None, render_args=None):
    if render_args is None:
        render_args = {
            "mode": "topdown",
            "screen_record": True,
            "window": False,
            "screen_size": (600, 600),
            "camera_position": (50, 50),
        }
    if trial_name is None:

        now = datetime.now()
        trial_name = now.strftime("%Y%m%d-%H%M%S")
    os.makedirs(trial_name, exist_ok=True)
    # evaluation
    total_reward = 0
    env = create_env(meta_drive_env_dict=meta_drive_env_dict)
    obs, _ = env.reset()
    try:
        for i in range(episodes):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            ret = env.render(
                **render_args
                # mode="topdown",
                #             screen_record=True,
                #             window=False,
                #             screen_size=(600, 600),
                #             camera_position=(50, 50)
            )
            if done:
                print("episode_reward", total_reward)
                break
        gif_name = os.path.join(trial_name, "demo.gif")
        env.top_down_renderer.generate_gif(gif_name)
    except Exception as e:
        print(e)
    finally:
        env.close()
    print("gif generation is finished ...")
    return Image(open(gif_name)), total_reward


def execute_trial(
    model,
    meta_drive_env_dict,
    trial_name,
    episodes,
    ppo_args,
    learn_args,
    verbose=True,
    parallel_envs=1,
):

    if verbose:
        # set log level to info
        # logging.basicConfig(level=logging.INFO)
        preview_env(meta_drive_env_dict)

    logging.info(f"starting execution of trial {trial_name}")
    logging.info("configs: ")
    logging.info(f"model: {model}")
    logging.info(f"meta_drive_env_dict: {meta_drive_env_dict}")
    logging.info(f"trial_name: {trial_name}")
    logging.info(f"episodes: {episodes}")
    logging.info(f"ppo_args: {ppo_args}")
    logging.info(f"learn_args: {learn_args}")
    # logging.info("verbose: ", verbose)
    logging.info("starting training")
    model = train(model, meta_drive_env_dict, ppo_args, learn_args, parallel_envs=4)
    logging.info("training is finished; evaluating model.")
    gif, total_reward = evaluate(
        model,
        episodes=episodes,
        meta_drive_env_dict=meta_drive_env_dict,
        trial_name=trial_name,
    )
    if verbose:
        logging.info(total_reward)
        try:
            gif.show()
        except Exception as e:
            print(e)
    model.save(os.path.join(trial_name, "model"))
    return model, gif, total_reward
