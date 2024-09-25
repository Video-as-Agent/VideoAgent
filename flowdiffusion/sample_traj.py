import os
import random
import shutil
from collections import defaultdict

def sample_trajectories(base_path, output_path, trajectories_per_task=100):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Scan the dataset structure
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task)
        if not os.path.isdir(task_path):
            continue

        print(f"Processing task: {task}")

        # Gather all trajectories for this task
        all_trajectories = []
        for corner in os.listdir(task_path):
            corner_path = os.path.join(task_path, corner)
            if not os.path.isdir(corner_path):
                continue
            
            for trajectory in os.listdir(corner_path):
                if trajectory.startswith('trajectory_'):
                    all_trajectories.append((corner, trajectory))

        # Sample trajectories
        sampled_trajectories = random.sample(all_trajectories, min(trajectories_per_task, len(all_trajectories)))

        # Copy sampled trajectories to new structure
        for i, (corner, trajectory) in enumerate(sampled_trajectories):
            src_path = os.path.join(task_path, corner, trajectory)
            dst_path = os.path.join(output_path, task, f"trajectory_{i}")
            shutil.copytree(src_path, dst_path)

        print(f"Sampled {len(sampled_trajectories)} trajectories for {task}")

    print("Sampling complete!")

# Usage
base_path = "/home/ubuntu/achint/datasets/mw_finetune_4/metaworld_dataset"
output_path = "/home/ubuntu/achint/datasets/mw_finetune_online_iter3"
sample_trajectories(base_path, output_path)