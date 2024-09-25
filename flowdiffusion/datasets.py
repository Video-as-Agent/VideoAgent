from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import imageio
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
import re
# from vidaug import augmentors as va

random.seed(0)

### Sequential Datasets: given first frame, predict all the future frames

class SequentialDatasetNp(Dataset):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(os.path.join(path, "**/out.npy"), recursive=True)
        if debug:
            sequence_dirs = sequence_dirs[:10]
        self.sequences = []
        self.tasks = []
    
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)

        self.sequences = obss
        self.tasks = tasks
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("training_samples: ", len(self.sequences))
        print("Done")

    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        task = seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
        return outputs, [task] * len(outputs)

    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        images = [self.transform(Image.fromarray(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task
        
class SequentialDataset(SequentialDatasetNp):
    def __init__(self, path="../datasets/frederik/berkeley", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = get_paths(path)
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(seq_dir))
            if len(seq) > 1:
                self.sequences.append(seq)
            task = seq_dir.split('/')[-6].replace('_', ' ')
            self.tasks.append(task)
        self.sample_per_seq = sample_per_seq
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        images = [self.transform(Image.open(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task

class SequentialDatasetVal(SequentialDataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = sorted([d for d in os.listdir(path) if "json" not in d], key=lambda x: int(x))
        self.sample_per_seq = sample_per_seq
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(os.path.join(path, seq_dir)))
            if len(seq) > 1:
                self.sequences.append(seq)
            
        with open(os.path.join(path, "valid_tasks.json"), "r") as f:
            self.tasks = json.load(f)
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Markovian datasets: given current frame, predict the next frame
class MarkovianDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = torch.FloatTensor(samples[start_ind].transpose(2, 0, 1) / 255.0)
        x = torch.FloatTensor(samples[start_ind+1].transpose(2, 0, 1) / 255.0)
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(samples[0].transpose(2, 0, 1) / 255.0)
    
class MarkovianDatasetVal(SequentialDatasetVal):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = self.transform(Image.open(samples[start_ind]))
        x = self.transform(Image.open(samples[start_ind+1]))
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(Image.open(samples[0]))
        
class AutoregDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        pred_idx = np.random.randint(1, len(samples))
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.cat(images[:-1], dim=0)
        x_cond[:, 3*pred_idx:] = 0.0
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
        
class AutoregDatasetNpL(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        N = len(samples)
        h, w, c = samples[0].shape
        pred_idx = np.random.randint(1, N)
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.zeros((N-1)*c, h, w)
        x_cond[(N-pred_idx-1)*3:] = torch.cat(images[:pred_idx])
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
    
# SSR datasets
class SSRDatasetNp(SequentialDatasetNp):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128), in_size=(48, 64), cond_noise=0.2):
        super().__init__(path, sample_per_seq, debug, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.fromarray(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.fromarray(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class SSRDatasetVal(SequentialDatasetVal):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), in_size=(48, 64)):
        print("Preparing dataset...")
        super().__init__(path, sample_per_seq, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.open(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.open(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class MySeqDatasetMW(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0513", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("-", " "))
        
        
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Randomly sample, from any intermediate to the last frame
# included_tasks = ["door-open", "door-close", "basketball", "shelf-place", "button-press", "button-press-topdown", "faucet-close", "faucet-open", "handle-press", "hammer", "assembly"]
# included_idx = [i for i in range(5)]
from torchvision import transforms
from PIL import Image

# class SequentialDatasetv2(Dataset):
#     def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False, device = 'cuda:0'):
#         print("Preparing dataset...")
#         self.device = torch.device(device)
#         self.sample_per_seq = sample_per_seq
#         self.frame_skip = frameskip

#         sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
#         self.transform = self.build_transforms(randomcrop, target_size)
#         self.tasks = []
#         self.sequences = []
#         self.x_2_store = []  # Store for x_2 values
#         self.task_store = []  # Store for task values
        
#         for seq_dir in sequence_dirs:
#             task = seq_dir.split("/")[-4]
#             seq_id = int(seq_dir.split("/")[-2])
#             seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
#             if seq:
#                 self.sequences.append(seq)
#                 initial_x_2 = torch.stack([self.transform(Image.open(seq[0])) for _ in range(7)], dim=0).permute(1, 0, 2, 3).to(self.device)
#                 self.x_2_store.append(initial_x_2.clone().detach())  # Clone and detach to ensure no computation graph
#                 self.tasks.append(task.replace("-", " "))
#                 self.task_store.append(task.replace("-", " "))  # Initialize with default task
                
#         print("Dataset prepared with", len(self.sequences), "sequences.")

#     def build_transforms(self, randomcrop, target_size):
#         # custom_crop = transforms.Lambda(lambda img: img.crop((80, 0, 280, 160)))
#         if randomcrop:
#             return transforms.Compose([
#                 # custom_crop,
#                 transforms.CenterCrop((160, 160)),
#                 transforms.RandomCrop((128, 128)),
#                 transforms.Resize(target_size),
#                 transforms.ToTensor()
#             ])
#         else:
#             return transforms.Compose([
#                 # custom_crop,
#                 transforms.CenterCrop((128, 128)),
#                 transforms.Resize(target_size),
#                 transforms.ToTensor()
#             ])

#     def get_samples(self, idx):
#         seq = self.sequences[idx]
#         if self.frame_skip is None:
#             start_idx = random.randint(0, len(seq) - self.sample_per_seq)
#         else:
#             start_idx = random.randint(0, len(seq) - self.frame_skip * self.sample_per_seq)

#         samples = [seq[start_idx + i * (self.frame_skip or 1)] for i in range(self.sample_per_seq) if start_idx + i * (self.frame_skip or 1) < len(seq)]
#         return samples

#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         try:
#             samples = self.get_samples(idx)
#             images = torch.stack([self.transform(Image.open(s)) for s in samples], dim=1).to(self.device)
#             x = rearrange(images[:, 1:], "c f h w -> (f c) h w")
#             x_cond = images[:, 0] # first frame
#             task = self.tasks[idx]
#             task_update = self.task_store[idx]
#             x_2 = self.x_2_store[idx]
#             return idx, x, x_cond, x_2, task, task_update
#         except Exception as e:
#             print(f"Error processing index {idx}: {e}")
#             return self.__getitem__((idx + 1) % len(self))

#     def update_x_2(self, idx, new_x_2):
#         if new_x_2.shape != self.x_2_store[idx].shape:
#             new_x_2 = new_x_2.reshape(self.x_2_store[idx].shape)
#         self.x_2_store[idx] = new_x_2.to(self.device).clone().detach()  # Ensure correct device # Clone and detach to avoid keeping references to the computation graph

#     def update_task(self, idx, new_task):
#         self.task_store[idx] = new_task  # Update task for the given index
        
class SequentialDatasetv2(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False, device = 'cuda:0'):
        print("Preparing dataset...")
        self.device = torch.device(device)
        self.sample_per_seq = sample_per_seq
        self.frame_skip = frameskip

        # sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        # trajectory_dirs = glob(f"{path}/**/*trajectory*/", recursive=True)
        # trajectory_dirs = glob(f"{path}/*/trajectory_*")
        trajectory_dirs = glob(f"{path}/*/*/*/trajectory_*")
        # print(trajectory_dirs)
        print(f"Found {len(trajectory_dirs)} trajectory directories")
        self.transform = self.build_transforms(randomcrop, target_size)
        self.video_transform = self.build_video_transforms(target_size)
        self.tasks = []
        # self.sequences = [d]
        self.feedback_store = {}  # Store for feedback
        self.trajectories = []
        self.x_2_store = {}  # Detailed store for x_2 values
        self.task_store = {}
        
        for dir in trajectory_dirs:
            # images = sorted(glob(f"{dir}*.png"), key=self.numerical_sort)
            # video = sorted(glob(f"{dir}*_output.gif"), key=self.numerical_sort)
            # images = sorted(glob(os.path.join(dir, "[0-9][0-9].png")), key=self.numerical_sort)
            # video = glob(os.path.join(dir, "frame_000_iter2_(305-308)_output_last.gif"))
            images = sorted(glob(os.path.join(dir, "*[0-9]*.png")), key=self.numerical_sort)
            video = glob(os.path.join(dir, "[0-9][0-9]_output.gif"))
            # print(images)
            # video_pattern = os.path.join(dir, '[0-9][0-9]binary_iter_1_checkpoint_305_output.gif')
            # video = sorted(glob(video_pattern), key=self.numerical_sort)
            if images and video:
                self.trajectories.append((images, video[0]))  # Assuming only one video per trajectory
                # print(dir)
                task = dir.split("/")[-2]  # Adjust depending on your directory structure
                # print(task)
                self.tasks.append(task.replace("-", " "))
                # Read feedback.txt and get the second line
                # Replace the existing feedback processing code with this:
                feedback_file_path = os.path.join(dir, 'feedback_long.txt')
                if os.path.exists(feedback_file_path):
                    with open(feedback_file_path, 'r') as feedback_file:
                        feedback_lines = feedback_file.readlines()
                        if feedback_lines:
                            last_line = feedback_lines[-1].strip()  # Get the last line of feedback
                            feedback_parts = last_line.split(":", 1)
                            if len(feedback_parts) > 1 and feedback_parts[0].strip().lower().startswith("base model"):
                                feedback = feedback_parts[1].strip()
                            else:
                                feedback = last_line
                        else:
                            feedback = "No feedback"
                else:
                    feedback = "No feedback"

                self.feedback_store[dir] = feedback
                    
                # if os.path.exists(feedback_file_path):
                #     with open(feedback_file_path, 'r') as feedback_file:
                #         feedback_lines = feedback_file.readlines()
                    
                #     processed_feedback = []
                #     for line in feedback_lines:
                #         # Split each line by ":" and take everything after it
                #         feedback_parts = line.split(":", 1)
                #         if len(feedback_parts) > 1 and feedback_parts[0].strip().lower() == "base model":
                #             processed_feedback.append(feedback_parts[1].strip())
                #         else:
                #             processed_feedback.append(line.strip())
                    
                #     feedback = "\n".join(processed_feedback) if processed_feedback else "No feedback"
                # else:
                #     feedback = "No feedback"

                # self.feedback_store[dir] = feedback
        
        print("Dataset prepared with", len(self.trajectories), "trajectories.")

    def numerical_sort(self, value):
        """ Extract numbers for sorting filenames numerically. """
        parts = re.findall(r'\d+', os.path.basename(value))
        return int(parts[0]) if parts else 0
    
    # def numerical_sort(self, value):
    #     """ Extract numbers for sorting filenames numerically. """
        # parts = re.findall(r'\d+', value)
        # return int(parts[0]) if parts else 0
        # match = re.search(r'frame_(\d+)', os.path.basename(value))
        # return int(match.group(1)) if match else 0
    
    def build_video_transforms(self, target_size=(320, 240)):
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def build_transforms(self, randomcrop, target_size):
        # custom_crop = transforms.Lambda(lambda img: img.crop((80, 0, 280, 160)))
        if randomcrop:
            return transforms.Compose([
                # custom_crop,
                transforms.CenterCrop((160, 160)),
                transforms.RandomCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                # custom_crop,
                transforms.CenterCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        # Return images, video, task, feedback
        images, video_path = self.trajectories[idx]
        # task = self.tasks[idx]
        sorted_images = sorted(images, key=self.numerical_sort)
        # print("Idx:", idx)
        # print("Images", sorted_images)
        # print("video:", video_path)
        feedback = self.feedback_store.get(os.path.dirname(images[0]), "No feedback")
        # print(feedback)

        # Load all images and transform them
        images_tensor = torch.stack([self.transform(Image.open(img)) for img in sorted_images]).to(self.device)
        # print("Image tensor:", images_tensor.shape)
        x_cond = images_tensor[0,:,:,:]
        # print("x_cond shape:", x_cond.shape)
        images_tensor = images_tensor[1:] 
        # Load video as a tensor
        video = imageio.mimread(video_path)
        video_tensor = torch.stack([
            self.video_transform(Image.fromarray(frame))  # Create a PIL image from each numpy array and apply the transform
            for frame in video  # Process all frames
        ]).to(self.device)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = video_tensor[1:]
        # print("video_tensor", video_tensor.shape)

        return idx, images_tensor, x_cond, feedback, video_tensor

class SequentialFlowDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.flows = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            flows = sorted(glob(f"{seq_dir}flow/*.npy"))
            self.sequences.append(seq)
            self.flows.append(np.array([np.load(flow) for flow in flows]))
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        return seq[0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # try:
            s = self.get_samples(idx)
            x_cond = self.transform(Image.open(s)) # [c f h w]
            x = rearrange(torch.from_numpy(self.flows[idx]), "f w h c -> (f c) w h") / 128
            task = self.tasks[idx]
            return x, x_cond, task
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(idx + 1 % self.__len__()) 

class SequentialNavDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=8, target_size=(64, 64), device = 'cuda:0'):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq
        self.device = torch.device(device)

        trajectory_dirs = glob(f"{path}/**/*trajectory*/", recursive=True)
        print(f"Found {len(trajectory_dirs)} trajectory directories")
        self.tasks = []
        self.trajectories = []
        self.sequences = []
        
        for dir in trajectory_dirs:
            # images = sorted(glob(os.path.join(dir, "[0-9][0-9].png")), key=self.numerical_sort)
            images = sorted(glob(os.path.join(dir, "[0-9][0-9].png")), key=self.numerical_sort)
            video = glob(os.path.join(dir, "[0-9][0-9]_output.gif"))
            if images and video:
                self.trajectories.append((images, video[0]))  # Assuming only one video per trajectory
                task = dir.split("/")[-3]  # Adjust depending on your directory structure
                self.tasks.append(task.replace("-", " "))
        
        print("Dataset prepared with", len(self.trajectories), "trajectories.")
        self.transform = video_transforms.Compose([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])
        # self.transform = transforms.Compose([
        #     transforms.Resize(target_size),
        #     transforms.ToTensor()
        # ])
        # self.video_transform = video_transforms.Compose([
        #     video_transforms.Resize(target_size)
        # ])
        print("Done")

    def numerical_sort(self, value):
        """ Extract numbers for sorting filenames numerically. """
        parts = re.findall(r'\d+', os.path.basename(value))
        return int(parts[0]) if parts else 0
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        images, video_path = self.trajectories[idx]
        sorted_images = sorted(images, key=self.numerical_sort)
        # print("Idx:", idx)
        # print("Images", images)
        # images_tensor = self.transform([Image.open(s) for s in images]).to(self.device) # [c f h w]
        # # x_cond = images_tensor[:, 0] # first frame
        # # x = rearrange(images_tensor[:, 1:], "c f h w -> (f c) h w") # all other frames
        
        # # Load video as a tensor
        # video = imageio.mimread(video_path)
        # video_tensor = self.transform([Image.fromarray(frame)  # Create a PIL image from each numpy array and apply the transform
        #     for frame in video]).to(self.device)
        # video_tensor = video_tensor.float() / 255.0
        # print(video_tensor.shape)
        
        
        # Process images
        image_frames = [np.array(Image.open(s).convert('RGB')) for s in sorted_images]
        images_tensor = self.transform(image_frames).to(self.device)  # [c f h w]
        # images_tensor = torch.stack([self.transform(Image.open(s)) for s in images]).to(self.device)  # [f c h w]
        
        # # Process video
        # video = imageio.mimread(video_path)
        # video_tensor = torch.stack([self.transform(Image.fromarray(frame)) for frame in video]).to(self.device)
        # video_tensor = video_tensor.float() / 255.0
        print(images_tensor.shape)
        images_tensor = rearrange(images_tensor, 'c f h w -> f c h w')
        x_cond = images_tensor[0,:,:,:]
        images_tensor = images_tensor[1:]
        # Process video
        video_frames = imageio.mimread(video_path)
        video_tensor = self.transform(video_frames).to(self.device)
        video_tensor = video_tensor.float() / 255.0
        
        # Ensure both tensors have the same number of frames
        images_tensor = rearrange(images_tensor,'f c h w -> c f h w')
        min_frames = min(images_tensor.size(1), video_tensor.size(1))
        # print("min_frames:", min_frames)
        images_tensor = images_tensor[:min_frames]
        video_tensor = video_tensor[1:]
        
        
        # task = self.tasks[self.frameid2seqid[idx]]
        task = self.tasks[idx]
        # print(images_tensor.shape)
        return idx, images_tensor, x_cond, task, video_tensor

class MySeqDatasetReal(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0606/processed_data", sample_per_seq=7, target_size=(48, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/*/*/", recursive=True)
        print(f"found {len(sequence_dirs)} sequences")
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*.png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("_", " "))
        
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")


if __name__ == "__main__":
    dataset = SequentialNavDataset("../datasets/thor")
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

