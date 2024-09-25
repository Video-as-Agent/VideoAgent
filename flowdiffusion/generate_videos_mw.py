from torchvision import transforms
import imageio
from feedback_binary_rf import chat_with_openai_rf
# from some_module import setup_trainer, numerical_sort
from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import os 
import torch.nn as nn
import torch.nn.functional as F
import json
from PIL import Image
import tqdm
import argparse
from einops import rearrange
import re

def setup_trainer(args, device, target_size=(128, 128)):
    # Ensure the model and components are set to run on the specified device
    unet = Unet().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3*7,
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[None],
        valid_set=[None],
        train_lr=1e-4,
        train_num_steps =70000,
        save_and_sample_every =210,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =4,
        valid_batch_size =4,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder ='/home/ubuntu/achint/models/mw_without_feedback/mw',
        fp16 =True,
        amp=True,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)

    return trainer

def process_batch(args, batch_images, texts, trainer, device):
    # custom_crop = transforms.Lambda(lambda img: img.crop((80, 0, 280, 160)))
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
    ])
    outputs = []
    gif_paths = []

    for img_path in batch_images:
        # Generate the expected output GIF path
        gif_path = os.path.splitext(img_path)[0] + f'_video_cond_base_305.gif'
        print(gif_path)
        # Check if the GIF already exists
        if os.path.exists(gif_path):
            print(f"Skipping existing GIF: {gif_path}")
            gif_paths.append(gif_path)
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        output = trainer.sample(img_tensor, [texts[batch_images.index(img_path)]], 1).cpu()  # Adjust parameters as needed
        output = output[0].reshape(-1, 3, *(128,128))
        print(output.shape)
        output_images = (output.numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        output_images = [np.array(Image.fromarray(frame).resize((320, 240))) for frame in output_images]
        imageio.mimsave(gif_path, output_images, duration=200, loop=1000)
        gif_paths.append(gif_path)
        print(f"Generated new GIF: {gif_path}")

    return gif_paths

def numerical_sort(value):
    """ Helper function to extract numbers for sorting filenames numerically. """
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='inference', choices=['train', 'inference'])
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None)
    parser.add_argument('-n', '--sample_steps', type=int, default=100)
    parser.add_argument('-g', '--guidance_weight', type=int, default=0)
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Set CUDA device by index')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)
    trainer = setup_trainer(args, device)

    base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset'  # Correct the path as necessary
    directories = sorted(os.listdir(base_path))
    # print(directories)
    midpoint = len(directories) // 2
    directories_to_process = directories
    print(directories_to_process)

    all_image_paths = []
    all_video_paths = []
    all_text_prompts = []

    for dir_name in directories_to_process:
        dir_path = os.path.join(base_path, dir_name)
        for subdir_name in sorted(os.listdir(dir_path)):
            subdir_path = os.path.join(dir_path, subdir_name)
            for corner_name in sorted(os.listdir(subdir_path)):
                corner_path = os.path.join(subdir_path, corner_name)
                for trajectory_name in sorted(os.listdir(corner_path)):
                    # print(trajectory_name)
                    trajectory_path = os.path.join(corner_path, trajectory_name)
                    if "trajectory" in trajectory_name:
                        trajectory_dir = trajectory_path
                        image_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.png')]
                        sorted_files = sorted(image_files, key=numerical_sort)
                        if sorted_files:
                            initial_frame_path = os.path.join(trajectory_dir, sorted_files[0])
                            environment_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(trajectory_dir))))

                            all_image_paths.append(initial_frame_path)
                            all_text_prompts.append(environment_name)

                            if len(all_image_paths) == 5:
                                # print(all_image_paths)
                                batch_video_paths = process_batch(args, all_image_paths, all_text_prompts, trainer, device)
                                all_video_paths.extend(batch_video_paths)
                                all_image_paths = []
                                all_text_prompts = []

    # Process any remaining images
    if all_image_paths:
        batch_video_paths = process_batch(args, all_image_paths, all_text_prompts, trainer, device)
        all_video_paths.extend(batch_video_paths)

    # # Generate feedback after all videos are processed
    # feedback_responses = chat_with_openai(all_image_paths, all_video_paths, all_text_prompts)
    
    # # Save feedback in text files within each trajectory directory
    # for idx, path in enumerate(all_image_paths):
    #     trajectory_dir = os.path.dirname(path)
    #     feedback_file_path = os.path.join(trajectory_dir, 'feedback.txt')
    #     with open(feedback_file_path, 'w') as feedback_file:
    #         formatted_feedback = f"{all_text_prompts[idx]}, {feedback_responses[idx]}"
    #         feedback_file.write(formatted_feedback)
    #     print(f"Feedback written to {feedback_file_path}")

if __name__ == "__main__":
    main()