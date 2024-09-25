import os
import re

def rename_gifs(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('base_output.gif'):
                match = re.match(r'frame_\d{3}base_output\.gif', filename)
                if match:
                    old_path = os.path.join(root, filename)
                    new_filename = 'frame_000base_output.gif'
                    new_path = os.path.join(root, new_filename)
                    
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f'Renamed: {old_path} -> {new_path}')

# Directory containing the dataset
base_dir = '/home/ubuntu/achint/datasets/mw_finetune_success/metaworld_dataset'

# Run the renaming process
rename_gifs(base_dir)

print("GIF renaming complete.")