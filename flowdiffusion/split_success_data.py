import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def adaptive_sampling(frames, start_frame_idx, total_frames, sample_per_seq=7, sampling_type='start_dense'):
    samples = [frames[start_frame_idx]]  # Always include the start frame
    remaining_frames = total_frames - start_frame_idx - 1
    
    if remaining_frames < sample_per_seq - 1:
        return samples + frames[start_frame_idx+1:start_frame_idx+sample_per_seq]
    
    if sampling_type == 'start_dense':
        intervals = np.linspace(1, remaining_frames/(sample_per_seq-1), sample_per_seq-1)**2
    elif sampling_type == 'end_dense':
        intervals = (remaining_frames/(sample_per_seq-1) - np.linspace(remaining_frames/(sample_per_seq-1), 1, sample_per_seq-1))**2
    else:  # 'uniform'
        intervals = np.linspace(0, remaining_frames, sample_per_seq-1)
    
    intervals = [int(i) for i in intervals]
    
    for interval in intervals:
        next_frame_idx = min(start_frame_idx + interval, total_frames - 1)
        samples.append(frames[next_frame_idx])
    
    return samples[:sample_per_seq]

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    sampling_types = ['start_dense', 'end_dense', 'uniform']
    trajectory_count = 0
    
    for start_idx in range(0, total_frames - 10, 3):  # Step by 3 to reduce total number of trajectories
        for sampling_type in sampling_types:
            samples = adaptive_sampling(frames, start_idx, total_frames, sampling_type=sampling_type)
            
            trajectory_dir = os.path.join(output_path, f"trajectory_{trajectory_count}")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            for i, frame in enumerate(samples):
                cv2.imwrite(os.path.join(trajectory_dir, f"frame_{i:03d}.png"), frame)
            
            trajectory_count += 1

    return trajectory_count

def process_task_videos(task_dir, output_base_path, simplified_task_name):
    output_task_path = os.path.join(output_base_path, simplified_task_name)
    os.makedirs(output_task_path, exist_ok=True)

    total_trajectories = 0
    for video_file in tqdm(os.listdir(task_dir), desc=f"Processing {simplified_task_name}"):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(task_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_video_path = os.path.join(output_task_path, video_name)
            os.makedirs(output_video_path, exist_ok=True)
            total_trajectories += process_video(video_path, output_video_path)
    
    print(f"Total trajectories for {simplified_task_name}: {total_trajectories}")

def main():
    base_dir = "/home/ubuntu/sreyas/Sreyas/AVDC_experiments/results/iteration_1_(305-308_reg)_video_cond_vlm/videos"
    output_base_path = "/home/ubuntu/achint/datasets/mw_finetune_4/metaworld_dataset"

    tasks = [
        "assembly-v2-goal-observable",
        "basketball-v2-goal-observable",
        "button-press-v2-goal-observable",
        "button-press-topdown-v2-goal-observable",
        "door-close-v2-goal-observable",
        "door-open-v2-goal-observable",
        "faucet-close-v2-goal-observable",
        "faucet-open-v2-goal-observable",
        "hammer-v2-goal-observable",
        "handle-press-v2-goal-observable",
        "shelf-place-v2-goal-observable"
    ]

    for task in tasks:
        task_dir = os.path.join(base_dir, task, "success")
        simplified_task_name = task.split('-v2-goal-observable')[0]
        if os.path.exists(task_dir):
            process_task_videos(task_dir, output_base_path, simplified_task_name)
        else:
            print(f"Directory not found: {task_dir}")

if __name__ == "__main__":
    main()