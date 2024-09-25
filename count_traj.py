import os
from collections import defaultdict

def count_trajectories(base_dir):
    total_trajectories = 0
    trajectories_per_task = defaultdict(int)

    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if os.path.isdir(task_path):
            for trajectory in os.listdir(task_path):
                trajectory_path = os.path.join(task_path, trajectory)
                if os.path.isdir(trajectory_path):
                    # Check if the directory contains PNG files
                    if any(file.endswith('.png') for file in os.listdir(trajectory_path)):
                        total_trajectories += 1
                        trajectories_per_task[task] += 1

    return total_trajectories, dict(trajectories_per_task)

# Directory containing the dataset
base_dir = '/home/ubuntu/achint/datasets/mw_finetune_success/metaworld_dataset'

# Count the trajectories
total, per_task = count_trajectories(base_dir)

print(f"Total number of trajectories: {total}")
print("\nTrajectories per task:")
for task, count in per_task.items():
    print(f"{task}: {count}")