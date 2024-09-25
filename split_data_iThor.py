import os
import random
import shutil

def get_samples(self, idx):
    seq = self.sequences[idx]
    total_frames = len(seq)
     # Randomly select the initial frame as per the previous method
    start_idx = random.randint(0, total_frames - self.sample_per_seq)

    # Calculate adaptive intervals
    adaptive_intervals = []
    initial_dense_factor = 0.9  # Adjust this factor to control the density of initial sampling
    decrease_factor = 0.2        # This factor controls how quickly the interval size increases

    # Generating adaptive intervals
    for i in range(self.sample_per_seq - 1):
        if i < self.sample_per_seq // 2:  # First half uses denser sampling
            interval = max(1, int(initial_dense_factor * (total_frames - start_idx) / self.sample_per_seq))
        else:
            # Gradually increase the interval size
            initial_dense_factor += decrease_factor
            interval = max(1, int(initial_dense_factor * (total_frames - start_idx) / self.sample_per_seq))

        adaptive_intervals.append(interval)

    # Collect samples using the adaptive intervals
    samples = []
    current_index = start_idx
    for interval in adaptive_intervals:
        current_index += interval
        if current_index < total_frames:
            samples.append(seq[current_index])
        else:
            break  # Avoid indexing beyond the sequence length

    # Ensure the start frame is included
    samples.insert(0, seq[start_idx])

    return start_idx, samples


def adaptive_sampling(seq, start_frame_idx, last_frame_idx, total_frames, sample_per_seq=7):
    # samples = [seq[start_frame_idx]]  # start frame included
    # samples.append(seq[last_frame_idx])  # last frame included
    # interval = 1  # Start with the smallest interval

    # Calculate adaptive intervals
    adaptive_intervals = []
    initial_dense_factor = 0.9  # Adjust this factor to control the density of initial sampling
    decrease_factor = 0.2 
    
    for i in range(sample_per_seq - 1):
        if i < sample_per_seq // 2:  # First half uses denser sampling
            interval = max(1, int(initial_dense_factor * (total_frames - start_frame_idx) / sample_per_seq))
        else:
            # Gradually increase the interval size
            initial_dense_factor += decrease_factor
            interval = max(1, int(initial_dense_factor * (total_frames - start_frame_idx) / sample_per_seq))

        adaptive_intervals.append(interval)
    
    samples = []
    current_index = start_frame_idx
    for interval in adaptive_intervals:
        current_index += interval
        if current_index < total_frames:
            samples.append(seq[current_index])
        else:
            break  # Avoid indexing beyond the sequence length

    # Ensure the start frame is included
    samples.insert(0, seq[start_frame_idx])    
    
    # # Collect additional frames until we have 7, ensuring to leave space for the last frame
    # current_index = start_frame_idx
    # while len(samples) < sample_per_seq - 1:  # -1 because last frame is already included
    #     current_index += interval
    #     if current_index >= last_frame_idx:
    #         break
    #     samples.insert(-1, seq[current_index])  # Insert before the last frame
    #     interval += 1  # Increase interval to spread out the sampling

    return samples

def process_dataset(input_base_path, output_base_path):
    environments = os.listdir(input_base_path)
    for environment in environments:
        env_path = os.path.join(input_base_path, environment)
        output_env_path = os.path.join(output_base_path, environment)
        os.makedirs(output_env_path, exist_ok=True)
        seeds = os.listdir(env_path)
        # print(camera_positions)
        #for position in camera_positions:
            # position_path = os.path.join(env_path, position)
            # output_position_path = os.path.join(output_env_path, position)
            # os.makedirs(output_position_path, exist_ok=True)
            # seeds = os.listdir(position_path)
        # print(seeds)
        for seed in seeds:
            seed_path = os.path.join(os.path.join(env_path, seed),"frames")
            output_seed_path = os.path.join(output_env_path, seed)
            os.makedirs(output_seed_path, exist_ok=True)
            images = sorted([img for img in os.listdir(seed_path) if img.endswith('.png')])
            total_frames = len(images)
            trajectory_count = 0
            print(total_frames)
            # Processing groups of 3 frames up to half the total frames
            for i in range(0, max(total_frames-7,1), 3):
                group = images[i:i+3]
                if len(group) < 1:
                    continue
                start_idx = 0
                # print(start_idx)
                start_frame = group[start_idx]

                # Randomly select the last frame from the last 5 frames
                #last_group = images[-5:] if total_frames >= 5 else images[-total_frames:]
                last_frame_idx =  total_frames - 1

                samples = adaptive_sampling(images, i + start_idx, last_frame_idx, total_frames)

                # Create a directory for each trajectory
                trajectory_dir = os.path.join(output_seed_path, f"trajectory_{trajectory_count}")
                os.makedirs(trajectory_dir, exist_ok=True)
                trajectory_count += 1

                # Copy files to the new directory
                for img in samples:
                    src = os.path.join(seed_path, img)
                    dst = os.path.join(trajectory_dir, img)
                    shutil.copy(src, dst)
                    

input_base_path = '/u1/a2soni/scratch/AVDC/ithor/thor_dataset'
output_base_path = '/u1/a2soni/Sreyas/offline data/iThor_split'
process_dataset(input_base_path, output_base_path)


#Process
"""
    I fix the first frame to the first one in the running index and the last frame to the final frame.
    I perform adaptive sampling as achint had done for generating data
    The reason is that the episode length per episode in the iThor dataset is much smaller. Hence I decided to fix the first frame and the last frame.
    """