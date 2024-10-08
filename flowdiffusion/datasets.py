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
from torchvision import transforms
from PIL import Image

random.seed(0)

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
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
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


class SequentialDatasetv2(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False, device = 'cuda:0'):
        print("Preparing dataset...")
        self.device = torch.device(device)
        self.sample_per_seq = sample_per_seq
        self.frame_skip = frameskip

        trajectory_dirs = glob(f"{path}/*/trajectory_*")
        print(f"Found {len(trajectory_dirs)} trajectory directories")
        self.transform = self.build_transforms(randomcrop, target_size)
        self.video_transform = self.build_video_transforms(target_size)
        self.tasks = []
        self.feedback_store = {}
        self.trajectories = []
        self.x_2_store = {} 
        self.task_store = {}
        
        for dir in trajectory_dirs:
            images = sorted(glob(os.path.join(dir, "frame_[0-9][0-9][0-9].png")), key=self.numerical_sort)
            video = glob(os.path.join(dir, "frame_000_video_cond_long_feedback_306.gif"))
            if images and video:
                self.trajectories.append((images, video[0]))
                task = dir.split("/")[-2] 
                self.tasks.append(task.replace("-", " "))
                feedback_file_path = os.path.join(dir, 'feedback_long.txt')
                if os.path.exists(feedback_file_path):
                    with open(feedback_file_path, 'r') as feedback_file:
                        feedback_lines = feedback_file.readlines()
                        if feedback_lines:
                            last_line = feedback_lines[-1].strip()
                            feedback_parts = last_line.split(":", 1)
                            if len(feedback_parts) > 1 and feedback_parts[0].strip().lower().startswith("online suggestive"):
                                feedback = feedback_parts[1].strip()
                            else:
                                feedback = last_line
                        else:
                            feedback = "No feedback"
                else:
                    feedback = "No feedback"

                self.feedback_store[dir] = feedback
                    
        print("Dataset prepared with", len(self.trajectories), "trajectories.")

    def numerical_sort(self, value):
        """ Extract numbers for sorting filenames numerically. """
        match = re.search(r'frame_(\d+)', os.path.basename(value))
        return int(match.group(1)) if match else 0
    
    def build_video_transforms(self, target_size=(320, 240)):
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def build_transforms(self, randomcrop, target_size):
        if randomcrop:
            return transforms.Compose([
                transforms.CenterCrop((160, 160)),
                transforms.RandomCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        images, video_path = self.trajectories[idx]
        sorted_images = sorted(images, key=self.numerical_sort)
        feedback = self.feedback_store.get(os.path.dirname(images[0]), "No feedback")
        
        images_tensor = torch.stack([self.transform(Image.open(img)) for img in sorted_images]).to(self.device)
        x_cond = images_tensor[0,:,:,:]
        images_tensor = images_tensor[1:] 
        video = imageio.mimread(video_path)
        video_tensor = torch.stack([
            self.video_transform(Image.fromarray(frame)) 
            for frame in video
        ]).to(self.device)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = video_tensor[1:]
        
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
            x_cond = self.transform(Image.open(s))
            x = rearrange(torch.from_numpy(self.flows[idx]), "f w h c -> (f c) w h") / 128
            task = self.tasks[idx]
            return x, x_cond, task
        
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
            images = sorted(glob(os.path.join(dir, "[0-9][0-9].png")), key=self.numerical_sort)
            video = glob(os.path.join(dir, "[0-9][0-9]_output.gif"))
            if images and video:
                self.trajectories.append((images, video[0]))
                task = dir.split("/")[-3]
                self.tasks.append(task.replace("-", " "))
        
        print("Dataset prepared with", len(self.trajectories), "trajectories.")
        self.transform = video_transforms.Compose([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])
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
        image_frames = [np.array(Image.open(s).convert('RGB')) for s in sorted_images]
        images_tensor = self.transform(image_frames).to(self.device)  # [c f h w]
        
        images_tensor = rearrange(images_tensor, 'c f h w -> f c h w')
        x_cond = images_tensor[0,:,:,:]
        images_tensor = images_tensor[1:]
        
        # Process video
        video_frames = imageio.mimread(video_path)
        video_tensor = self.transform(video_frames).to(self.device)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = rearrange(video_tensor, 'c f h w -> f c h w')
        
        # Ensure both tensors have the same number of frames
        images_tensor = rearrange(images_tensor,'f c h w -> c f h w')
        video_tensor = video_tensor[1:]
        
        video_tensor = rearrange(video_tensor, 'f c h w -> c f h w')
        
        task = self.tasks[idx]
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

