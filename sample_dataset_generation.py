import os
import random
import shutil

def adaptive_sampling(seq, start_frame_idx, last_frame_idx, total_frames, sample_per_seq=8):
    adaptive_intervals = []
    initial_dense_factor = 0.9 
    decrease_factor = 0.2 
    
    total_interval = last_frame_idx - start_frame_idx

    for i in range(sample_per_seq - 1):
        if i < sample_per_seq // 2:
            interval = max(1, int(initial_dense_factor * (total_interval) / sample_per_seq))
        else:
            initial_dense_factor += decrease_factor
            interval = max(1, int(initial_dense_factor * (total_interval) / sample_per_seq))

        adaptive_intervals.append(interval)
    
    samples = []
    current_index = start_frame_idx
    for interval in adaptive_intervals:
        current_index += interval
        if current_index <= last_frame_idx:
            samples.append(seq[current_index])
        else:
            break  
        
    samples.insert(0, seq[start_frame_idx])    
    
    return samples


def process_dataset(input_base_path, output_base_path):
    environments = os.listdir(input_base_path)
    for environment in environments:
        env_path = os.path.join(input_base_path, environment)
        output_env_path = os.path.join(output_base_path, environment)
        os.makedirs(output_env_path, exist_ok=True)
        camera_positions = os.listdir(env_path)
        for position in camera_positions:
            position_path = os.path.join(env_path, position)
            output_position_path = os.path.join(output_env_path, position)
            os.makedirs(output_position_path, exist_ok=True)
            seeds = os.listdir(position_path)
            for seed in seeds:
                seed_path = os.path.join(position_path, seed)
                output_seed_path = os.path.join(output_position_path, seed)
                os.makedirs(output_seed_path, exist_ok=True)
                images = sorted([img for img in os.listdir(seed_path) if img.endswith('.png')], key=lambda x: int(x.split('.')[0]))
                total_frames = len(images)
                trajectory_count = 0

                for i in range(0, total_frames - 12, 3):
                    group = images[i:i+3]
                    if len(group) < 1:
                        continue
                    start_idx = random.randint(0, len(group)-1)
                    
                    last_group = images[-5:] if total_frames >= 5 else images[-total_frames:]
                    last_frame_idx = random.randint(total_frames - len(last_group), total_frames - 1)

                    samples = adaptive_sampling(images, i + start_idx, last_frame_idx, total_frames)

                    trajectory_dir = os.path.join(output_seed_path, f"trajectory_{trajectory_count}")
                    os.makedirs(trajectory_dir, exist_ok=True)
                    trajectory_count += 1

                    for img in samples:
                        src = os.path.join(seed_path, img)
                        dst = os.path.join(trajectory_dir, img)
                        shutil.copy(src, dst)

input_base_path = '/dataset/metaworld'
output_base_path = '/dataset/split_metaworld_dataset'
process_dataset(input_base_path, output_base_path)
