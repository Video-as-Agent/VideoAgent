import os
import re
from pathlib import Path

def rename_images(directory):
    for root, dirs, files in os.walk(directory):
        # Check if this directory contains PNG files
        if any(file.endswith('.png') for file in files):
            png_files = [f for f in files if f.endswith('.png')]
            
            # Check if files are already in the desired format
            if all(f.startswith('frame') for f in png_files):
                continue
            
            # Sort files numerically
            png_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
            
            # Rename files
            for i, old_name in enumerate(png_files):
                new_name = f'frame{i:03d}.png'
                old_path = os.path.join(root, old_name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')

# Directory containing the trajectories
base_dir = '/home/ubuntu/achint/datasets/mw_finetune_success/metaworld_dataset'

# Run the renaming process
rename_images(base_dir)

print("Image renaming complete.")