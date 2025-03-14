
class MetaDriveEnvOOR(MetaDriveEnv):
    def __init__(self, config, out_of_road_penalty=0.0):
        
        super().__init__(config)
        self.out_of_road_penalty = out_of_road_penalty

    def reward_function(self, vehicle_id: str):
        """
        Override the reward function to upweight the out-of-road penalty.
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Start with the default reward
        reward, base_reward_info = super().reward_function(vehicle_id)

        # Upweight the out-of-road penalty
        if self._is_out_of_road(vehicle):
            print(self.config["out_of_road_penalty"])
            print(self.out_of_road_penalty)
            penalty = -self.config["out_of_road_penalty"] * self.out_of_road_penalty
            print(reward)
            reward += penalty
            print(reward)
            step_info["out_of_road_penalty"] = penalty

        # Combine with base reward info
        step_info.update(base_reward_info)

        return reward, step_info
    
def create_custom_environment(difficulty, render=False, monitor=True):
    configs = {
        "easy": {"map": 3, "traffic_density": 0.1, "random_lane_width": False, "use_render": render},
        "medium": {"map": 5, "traffic_density": 0.3, "random_lane_width": True, "use_render": render},
        "hard": {"map": 7, "traffic_density": 0.5, "random_lane_width": True, "use_render": render},
    }

    # reward_weights = {
    #     "easy": {"safety": 1.0, "efficiency": 0.5, "comfort": 0.2, "rule_compliance": 0.3},
    #     "medium": {"safety": 1.0, "efficiency": 0.5, "comfort": 0.2, "rule_compliance": 0.3},
    #     "hard": {"safety": 1.0, "efficiency": 0.5, "comfort": 0.2, "rule_compliance": 0.3},
    # }
    
    base_env = MetaDriveEnvOOR(
        configs[difficulty],
        out_of_road_penalty=1.5,
    )

    if monitor:
        return Monitor(base_env)
    else:
        return base_env
    

def train_agent_with_transfer_custom(custom_env, timesteps=1000, model=None):
    logger = PerformanceLogger()
    if model is None:
        model = PPO("MlpPolicy", custom_env, verbose=2, n_steps=4096)
    else:
        print("Continuing training with transfer learning...")
    logger.reset()  # Reset logger for the new training phase
    model.learn(total_timesteps=timesteps, callback=logger)
    return model, logger


def curriculum_experiment_with_transfer_custom(difficulty_order, timesteps_per_difficulty=1000):
    results = []
    models = []
    model = None 
    for difficulty in difficulty_order:
        print(f"Training on {difficulty} difficulty (No Transfer)...")
        env = create_custom_environment(difficulty, render=False, monitor=False)
        try:
            model, logger = train_agent_with_transfer_custom(env, timesteps=timesteps_per_difficulty, model = model)
            results.append((difficulty, logger))
            models.append((difficulty, model))
        finally:
            env.close()

    # Plot performance for each difficulty
    for difficulty, logger in results:
        print(f"Performance for {difficulty} difficulty:")
        plot_performance(logger, f"Performance on {difficulty} (No Transfer)")
    return results,  models
  
results, models = curriculum_experiment_with_transfer_custom(['easy', 'medium', 'hard'], timesteps_per_difficulty=100_000)
model = models[2][1]
obs, _ = env.reset()
try:
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        # total_reward += reward
        ret = env.render(mode="topdown",
                         screen_record=True,
                         window=False,
                         screen_size=(600, 600),
                         camera_position=(50, 50))
        if done:
            # print("episode_reward", total_reward)
            break


finally:
    env.close()


from IPython.display import Image

Image(open("demo.gif", 'rb').read())


