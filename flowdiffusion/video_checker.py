import os

def count_trajectories_without_gif(base_path):
    missing_gif_count = 0
    total_trajectories = 0

    for root, dirs, files in os.walk(base_path):
        # Check if this is a trajectory directory
        if any(file.endswith('.gif') for file in files):
            total_trajectories += 1
            
            # Check for the specific GIF file
            gif_exists = any(file.endswith("binary_iter_3_(305_310)_checkpoint_315_output.gif") for file in files)
            
            if not gif_exists:
                missing_gif_count += 1
                print(f"Missing GIF in: {root}")

    print(f"Total trajectories: {total_trajectories}")
    print(f"Trajectories without the specific GIF: {missing_gif_count}")
    print(f"Percentage of trajectories missing the GIF: {(missing_gif_count / total_trajectories) * 100:.2f}%")

# Usage
base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset/'

count_trajectories_without_gif(base_path)