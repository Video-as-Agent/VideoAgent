import os
import shutil
from pathlib import Path
from collections import defaultdict

def reorganize_directories(source_dir1, source_dir2, target_dir):
    # Create the target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Dictionary to keep track of the highest trajectory number for each task
    trajectory_counters = defaultdict(int)

    def copy_trajectory(src, task):
        nonlocal trajectory_counters
        trajectory_counters[task] += 1
        new_traj_name = f"trajectory_{trajectory_counters[task]}"
        dst = os.path.join(target_dir, task, new_traj_name)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Process the first source directory
    for task in os.listdir(source_dir1):
        task_path = os.path.join(source_dir1, task)
        if os.path.isdir(task_path):
            for corner in os.listdir(task_path):
                corner_path = os.path.join(task_path, corner)
                if os.path.isdir(corner_path):
                    for seed in os.listdir(corner_path):
                        seed_path = os.path.join(corner_path, seed)
                        if os.path.isdir(seed_path):
                            for traj_dir in os.listdir(seed_path):
                                if traj_dir.startswith('trajectory_'):
                                    src = os.path.join(seed_path, traj_dir)
                                    copy_trajectory(src, task)

    # Process the second source directory
    for task in os.listdir(source_dir2):
        task_path = os.path.join(source_dir2, task)
        if os.path.isdir(task_path):
            for corner in os.listdir(task_path):
                if corner.startswith('corner_'):
                    corner_path = os.path.join(task_path, corner)
                    if os.path.isdir(corner_path):
                        for traj_dir in os.listdir(corner_path):
                            if traj_dir.startswith('trajectory_'):
                                src = os.path.join(corner_path, traj_dir)
                                copy_trajectory(src, task)

# Define the paths
source_dir1 = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset'
source_dir2 = '/home/ubuntu/achint/datasets/mw_finetune_balanced/metaworld_dataset'
target_dir = '/home/ubuntu/achint/datasets/mw_finetune_success/metaworld_dataset'

# Run the reorganization
reorganize_directories(source_dir1, source_dir2, target_dir)

print("Directory reorganization complete.")