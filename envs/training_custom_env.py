from metadrive.envs import MetaDriveEnv
import numpy as np

class CustomMetaDriveEnv(MetaDriveEnv):
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]

        safety = -sum(
            [np.linalg.norm(vehicle.position - obj.position) for obj in self.traffic_objects]
        ) - (5 if vehicle.crash_vehicle else 0)

        efficiency = -np.linalg.norm(vehicle.position - self.goal_position) - 0.1 * self.simulation_time

        comfort = np.abs(vehicle.velocity - vehicle.last_velocity) + np.abs(vehicle.steering_angle - vehicle.last_steering_angle)

        rule_compliance = (1 if vehicle.speed <= self.speed_limit else -1) + (1 if vehicle.respects_signals else -1)

        reward = 1.0 * safety + 0.5 * efficiency - 0.2 * comfort + 0.3 * rule_compliance
        return reward
