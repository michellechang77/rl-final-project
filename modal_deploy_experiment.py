import modal
from rl_final_proj_experiments import curriculum_experiment, upload_folder_to_gcs

# Import your existing functions and dependencies here
# Assume all required functions from your script are available

# Define the Modal stub
app = modal.App("rl-training-app")


# Define the image for your runtime environment
@app.function(
    image=(
        modal.Image.debian_slim()
        .apt_install("git", "python3-opencv")
        .run_commands(
            "git clone https://github.com/metadriverse/metadrive.git &&"
            "cd metadrive &&"
            "pip install -e . &&"
            "cd .."
        )
        .pip_install(
            [
                "stable-baselines3",
                "pandas",
                "matplotlib",
                "torch",
                "google-cloud-storage",
                "opencv-python",
            ]
        )
    ),
    secrets=[modal.Secret.from_name("googlecloud-secret")],
    timeout=50000,  # Extend timeout as needed
    gpu="L4",
)
def main():
    from google.oauth2 import service_account
    from google.cloud import storage
    import os
    import json

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    service_account_json_path = "/tmp/service_account.json"
    with open(service_account_json_path, "w", encoding="utf-8") as f:
        f.write(os.environ["SERVICE_ACCOUNT_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json_path
    # Redefine your main logic here
    time_steps = 200_000
    difficulty_order = [
        "hard",
        "medium",
        "easy",
    ]

    # Call your existing curriculum experiment and upload functions
    results, models, final_models = curriculum_experiment(
        difficulty_order,
        timesteps_per_difficulty=time_steps,
        transfer=True,
        trial_name="600k-hard-medium-easy",
    )

    upload_folder_to_gcs("trials", "snap-chef-recipe", "rl-final-project")


# Entry point for Modal
if __name__ == "__main__":
    with app.run():
        main()
