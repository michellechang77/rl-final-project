import matplotlib.pyplot as plt
import os

def log_rewards(rewards, labels, output_path="logs/reward_comparison.png"):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, rewards)
    plt.xlabel("Curriculum Strategy")
    plt.ylabel("Cumulative Reward")
    plt.title("Curriculum Learning Comparison")
    plt.savefig(output_path)
    plt.show()
