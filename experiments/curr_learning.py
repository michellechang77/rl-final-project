from stable_baselines3 import PPO


from envs.training_custom_env import CustomMetaDriveEnv

def train_agent(env, timesteps=50000, model_path=None):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    if model_path:
        model.save(model_path)
    env.close()

def run_curriculum():
    difficulties = {
        "easy": {"map": 3, "traffic_density": 0.1, "wheel_friction": 2.5},
        "medium": {"map": 5, "traffic_density": 0.3, "wheel_friction": 1.5},
        "hard": {"map": 7, "traffic_density": 0.5, "wheel_friction": 0.5},
    }

    # Run Increasing Difficulty Curriculum
    for difficulty, config in difficulties.items():
        print(f"Training on {difficulty} difficulty...")
        env = CustomMetaDriveEnv(config)
        model_path = f"models/ppo_{difficulty}.zip"
        train_agent(env, timesteps=50, model_path=model_path)

if __name__ == "__main__":
    run_curriculum()
