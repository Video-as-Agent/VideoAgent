import os
import argparse
from PIL import Image
from torchvision import transforms
import torch
from feedback_mw_extend import chat_with_openai
from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
import re
import os
import base64
import json
from tqdm import tqdm
import traceback
import warnings
warnings.filterwarnings('ignore')

# def numerical_sort(value):
#     """ Helper function to extract numbers for sorting filenames numerically. """
#     numbers = re.findall(r'\d+', value)
#     return int(numbers[0]) if numbers else 0

def numerical_sort(value):
    """ Extract numbers for sorting filenames numerically. """
    parts = re.findall(r'\d+', os.path.basename(value))
    return int(parts[0]) if parts else 0
    
def encode_gif(image_path):
    frame = Image.open(image_path)
    nframes = 0
    encoded = []
    
    # Ensure the cache directory exists
    cache_dir = "feedback_fewshot/.cache/"
    os.makedirs(cache_dir, exist_ok=True)
    
    while True:
        try:
            image_file = os.path.join(cache_dir, os.path.basename(image_path) + "-" + str(nframes) + ".jpg")
            frame.convert('RGB').save(image_file)
            with open(image_file, "rb") as image:
                encoded.append(base64.b64encode(image.read()).decode('utf-8'))
            nframes += 1
            if nframes == 8: break
            frame.seek(nframes)
        except EOFError:
            break
    
    return encoded

def is_valid_feedback(feedback):
    """ Check if the feedback is valid (contains a proper feedback). """
    return feedback.strip().split(', ')[-1] not in ["Feedback not generated. Please retry."]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--number', type=int, default=None)
    args = parser.parse_args()
    
    base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset'
    count = 0
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if "trajectory" in dir:
                trajectory_dir = os.path.join(root, dir)
                feedback_file_path = os.path.join(trajectory_dir, 'feedback_long.txt')

                # # Check if feedback.txt exists and if it contains valid feedback
                # if os.path.exists(feedback_file_path):
                #     with open(feedback_file_path, 'r') as feedback_file:
                #         feedback = feedback_file.read()
                    # if is_valid_feedback(feedback):
                    #     print(f"Skipping {trajectory_dir}, already has valid feedback.")
                    #     continue
                    # else:
                    #     print(f"Retrying feedback generation for {trajectory_dir} due to invalid feedback.")

                image_files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('.png')], key=numerical_sort)
                # video_files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('binary_iter_2__(304)_checkpoint_306_output.gif')])
                video_files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('video_cond_base_305.gif')], key = numerical_sort)

                if image_files and video_files:
                    initial_frame_path = os.path.join(trajectory_dir, image_files[0])
                    video_path = os.path.join(trajectory_dir, video_files[0])
                    environment_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(trajectory_dir))))
                    
                    # Check the number of frames in the GIF
                    try:
                        gif_frames = encode_gif(video_path)
                        if len(gif_frames) < 7:
                            print(f"Skipping {video_path}: less than 8 frames.")
                            continue
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        continue
                    
                    print(video_path)
                    print(initial_frame_path)
                    
                    # Generate feedback
                    feedback_response = chat_with_openai([initial_frame_path], [video_path], [environment_name])
                    formatted_feedback = f"{environment_name}, {feedback_response[0]}"

                    # Save feedback
                    with open(feedback_file_path, 'a') as feedback_file:
                        feedback_file.write('\n' + f"Base model_{args.number}:" + formatted_feedback)
                    print(f"Feedback generated and saved for {trajectory_dir}")
                    count = count + 1
    print("Total trajectories:", count)

if __name__ == "__main__":
    main()