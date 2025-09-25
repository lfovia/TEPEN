import torch
import torch.nn.functional as F
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat import Env, get_config
import numpy as np
import random
import matplotlib.pyplot as plt
import habitat
from config_utils import hm3d_data_config,hm3d_config
from constants import *
import open3d as o3d
import quaternion
from data_utils.geometry_tools import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18 as imagenet_resnet18
import time
import cv2
import json
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_agent_inputs(goal_image_batch, goal_mask_batch, episode_images_batch):
    B, T, C, H, W = episode_images_batch.shape
    episode_images_batch = episode_images_batch.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
    goal_image_batch = goal_image_batch.permute(0, 2, 3, 1)            # (B, H, W, C)
    goal_mask_batch = goal_mask_batch.permute(0, 2, 3, 1)              # (B, H, W, 1)
    return goal_mask_batch, goal_image_batch, episode_images_batch
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def normalize_coords(pixel_coords, width=160, height=120):
    x = (pixel_coords[..., 0] / width) 
    y = (pixel_coords[..., 1] / height) 
    return torch.stack([x, y], dim=-1)

class DepthPolicy(nn.Module):

    def __init__(self,max_token_length=64,device='cuda:1'):
        super().__init__()
        self.device = device
        self.max_token_length = max_token_length
        #
        self.depth_resnet=imagenet_resnet18()
        self.depth_resnet.conv1=nn.Conv2d(1,self.depth_resnet.conv1.out_channels,
                                             kernel_size=self.depth_resnet.conv1.kernel_size,
                                             stride=self.depth_resnet.conv1.stride,
                                             padding=self.depth_resnet.conv1.padding,
                                             bias=self.depth_resnet.conv1.bias)
        self.history_backbone = nn.Sequential(*(list(self.depth_resnet.children())[:-1]),nn.Flatten()).to(device)
        # goal backbone
        self.goal_backbone = imagenet_resnet18()
        self.goal_backbone.conv1 = nn.Conv2d(2,self.goal_backbone.conv1.out_channels,
                                             kernel_size=self.goal_backbone.conv1.kernel_size,
                                             stride=self.goal_backbone.conv1.stride,
                                             padding=self.goal_backbone.conv1.padding,
                                             bias=self.goal_backbone.conv1.bias)
        self.goal_backbone = nn.Sequential(*(list(self.goal_backbone.children())[:-1]),nn.Flatten()).to(device)
        # goal fusion, project the representations to all the input tokens
        self.goal_concat_proj = nn.Linear(512,256,device=device)
        # goal input token
        self.goal_input_proj = nn.Linear(512,768,device=device)
        # transformer-decoder policy
        self.dt_policy = nn.TransformerDecoder(nn.TransformerDecoderLayer(768,4,dropout=0.25,batch_first=True,device=device),4)
        self.po_embedding = nn.Embedding(max_token_length,768,device=device)
        nn.init.normal_(self.po_embedding.weight,0,0.01)
        # prediction heads, including policy head, tracking head and distance head
        self.action_head = nn.Linear(768,6,device=device)
        self.distance_head = nn.Linear(768,1,device=device)
        self.goal_head = nn.Linear(768,2,device=device)
    
    def forward(self,goal_mask,goal_image,episode_image):
        print(goal_mask.shape,goal_image.shape,episode_image.shape)
        # goal concat token shape = (B,1,256), goal input token shape = (B,1,256)
        goal_mask_tensor = torch.as_tensor(goal_mask/255.0,dtype=torch.float32,device=self.device).contiguous()#.permute(0,3,1,2).contiguous()
        goal_image_tensor = torch.as_tensor(goal_image/255.0,dtype=torch.float32,device=self.device).contiguous()#.permute(0,3,1,2).contiguous()

        goal_token = self.goal_backbone(torch.concat((goal_image_tensor,goal_mask_tensor),dim=1)).unsqueeze(1)

        goal_concat_token = self.goal_concat_proj(goal_token)
        goal_input_token = self.goal_input_proj(goal_token)  

        # history image token shape = (B,64,512), and the episode input tokens are concated to (B,64,512+256)
        episode_image_tensor = torch.as_tensor(episode_image/255.0,dtype=torch.float32,device=self.device).contiguous()#.permute(0,1,4,2,3).contiguous()
        B,T,C,H,W = episode_image_tensor.shape
        episode_image_tensor = episode_image_tensor.view(-1,C,H,W)
        epc_token = self.history_backbone(episode_image_tensor)
        epc_token = epc_token.view(B,T,epc_token.shape[-1])
        epc_token = torch.concat((epc_token,goal_concat_token.tile((1,epc_token.shape[1],1))),dim=-1)
        
        # add the position embedding
        pos_indice = torch.arange(self.max_token_length).expand(epc_token.shape[0],self.max_token_length).to(self.device)
        pos_embed = self.po_embedding(pos_indice)
        epc_token = epc_token + pos_embed
        tgt_mask = generate_square_subsequent_mask(self.max_token_length).to(self.device)
        out_token = self.dt_policy(tgt=epc_token,
                                   memory=goal_input_token,
                                   tgt_mask = tgt_mask)
        action_pred = self.action_head(out_token)
        # distance_pred = torch.sigmoid(self.distance_head(out_token))
        # # goal_pred = self.goal_head(out_token)
        # goal_pred = torch.sigmoid(self.goal_head(out_token)) 
        # # print(action_pred.shape,distance_pred.shape,goal_pred.shape)
        return out_token,action_pred 
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories_dir, max_len=64,mode="train"):
        self.max_len=max_len
        self.trajectory_paths=[]
        
        if mode == "train":
            self.trajectory_paths = [
                os.path.join(trajectories_dir, f)
                for f in os.listdir(trajectories_dir)
                if os.path.exists(os.path.join(trajectories_dir, f, "data.json"))
            ][:100000]
        elif mode == "val":
            count=0
            self.trajectory_paths = [
                os.path.join(trajectories_dir, f)
                for f in os.listdir(trajectories_dir)
                if os.path.exists(os.path.join(trajectories_dir, f, "data.json"))
            ][100000:110000]
            
        self.max_len = max_len
        self.transform = transforms.Compose([
            #transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.trajectory_paths)
    

    def __getitem__(self, idx):
        path = self.trajectory_paths[idx]
        with open(os.path.join(path, "data.json"), 'r') as f:
            data = json.load(f)
        goal_mask = self.transform(Image.open(os.path.join(path, "goal_mask.jpg")))
        goal_depth = self.transform(Image.open(os.path.join(path, "goal_depth.jpg")).convert("L"))
        
        depth_tensors = [self.transform(Image.open(img).convert("L")) for img in data["depth_path"]]


        T = len(depth_tensors)
        mask = torch.ones(self.max_len)
        if T < self.max_len:
            pad = self.max_len - T
            depth_tensors+=[torch.zeros_like(depth_tensors[0]) for _ in range(pad)]
            mask[T:] = 0

        depths = torch.stack(depth_tensors[:self.max_len])

        actions = torch.tensor(data["actions"], dtype=torch.long)
        distances = torch.tensor(data["distances"], dtype=torch.float32)
        distances = distances / 64
        track_coords = torch.tensor(data["track_point"], dtype=torch.float32)

        if T < self.max_len:
            pad = self.max_len - T
            actions = torch.cat([actions, torch.zeros(pad, dtype=torch.long)])
            distances = torch.cat([distances, torch.zeros(pad)])
            track_coords = torch.cat([track_coords, torch.zeros((pad, 2), dtype=torch.float32)])
        
        return  goal_mask, goal_depth, depths,actions, distances, track_coords, mask

        

def train_policy(model,train_dataset,val_dataset, device, batch_size=32, num_epochs=10):

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=100000)

    tr_losses=[]
    val_losses=[]
    ce_loss = nn.CrossEntropyLoss()
    # mse_loss = nn.MSELoss()
    # bce_loss = nn.BCELoss()
    for epoch in range(num_epochs):
            
        start=time.time()
        model.train()
        total_loss = 0
        action_loss=0
        dist_loss=0
        track_loss=0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            goal_mask,goal_depth,  episode_depth, oracle_actions, oracle_distance, oracle_goal , mask = [
                b.to(device) for b in batch
            ]
            
            _,a2= model(goal_mask, goal_depth, episode_depth)
            
            action_loss = ce_loss(a2.view(-1, 6), oracle_actions.view(-1))
            
            loss = action_loss # + 0.1* distance_loss + 1 * goal_loss
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader.dataset):.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f"/data/athira/Author_data/New/depth/depth_{epoch+1}.pth")
        print("Train completed in" , time.time()-start)
        file = open("/data/athira/Author_data/New/depth/l_log.txt", "a") 
        file.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(dataloader.dataset):.4f}\n")      
        file.write(f"Action Loss: {action_loss / len(dataloader.dataset):.4f}, Distance Loss: {dist_loss / len(dataloader.dataset):.4f},Train Loss: {track_loss / len(dataloader.dataset):.4f}\n")            
        file.close() 
        
        # Validate the model
        val_loss=val_policy(model, val_dataset, device, batch_size=4)
        tr_losses.append(total_loss / len(dataloader.dataset))
        val_losses.append(val_loss)
        
    plt.plot(tr_losses,label="train loss")
    plt.plot(val_losses,label="val loss")
    plt.legend()
    plt.savefig("/data/athira/Author_data/New/depth/l_losses.jpg")
    plt.close()
    

def val_policy(model, dataset, device, batch_size=32, max_tsne_samples=5000):
    """
    Validate the model on the dataset.
    Computes action loss, accuracy, and generates t-SNE visualization of features.

    Args:
        model: the policy model
        dataset: validation dataset
        device: cuda/cpu
        batch_size: batch size for dataloader
        max_tsne_samples: maximum number of samples to use for t-SNE
    """
    print("Starting validation...")
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4)
    model.eval()

    total_loss = 0
    total_accuracy = 0
    ce_loss = nn.CrossEntropyLoss()

    # Store features and labels for t-SNE
    features_all = []
    labels_all = []

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Unpack batch and move to device
            goal_mask, goal_depth, episode_depth, oracle_actions, oracle_distance, oracle_goal, mask = [
                b.to(device) for b in batch
            ]
            torch.cuda.empty_cache()

            # Forward pass
            token,action_logits = model(goal_mask, goal_depth, episode_depth)  # token: [B,T,D], action_logits: [B,T,6]

            # Flatten for loss computation: [B*T, num_classes]
            action_logits_flat = action_logits.view(-1, 6)
            oracle_actions_flat = oracle_actions.view(-1)

            # Compute action loss
            action_loss = ce_loss(action_logits_flat, oracle_actions_flat)
            total_loss += action_loss.item() * goal_mask.size(0)

            # Compute accuracy
            final_actions = torch.argmax(action_logits_flat, dim=-1)
            correct = (final_actions == oracle_actions_flat).sum().item()
            total_accuracy += correct

            # Save features for t-SNE: flatten (B*T, D)
            features_all.append(token.view(-1, token.shape[-1]).cpu())
            labels_all.append(oracle_actions_flat.cpu())

    # Stack features and labels
    features_all = torch.cat(features_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    # Remove NaNs/Infs
    finite_mask = torch.isfinite(features_all).all(dim=1)  # torch.bool
    X_clean = features_all[finite_mask].numpy()
    y_clean = labels_all[finite_mask].numpy()
    print(f"Kept {X_clean.shape[0]} / {features_all.shape[0]} samples after removing NaNs/Infs")

    # Optional: subsample to avoid t-SNE crash
    if X_clean.shape[0] > max_tsne_samples:
        idx = np.random.choice(X_clean.shape[0], max_tsne_samples, replace=False)
        X_clean = X_clean[idx]
        y_clean = y_clean[idx]

    # Replace any remaining NaNs/Infs
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)

    # Standardize features
    X_clean = (X_clean - X_clean.mean(axis=0, keepdims=True)) / (X_clean.std(axis=0, keepdims=True) + 1e-6)

    # Run t-SNE
    tsne = TSNE(n_components=2, init="random", perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(X_clean)

    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_clean, cmap="tab10", s=5)
    plt.legend(*scatter.legend_elements(), title="Actions")
    plt.title("t-SNE of Action Logits (cleaned)")
    plt.savefig("tsne_plot_authorD.png", dpi=300)
    plt.close()
    print("t-SNE plot saved at tsne_plot_authorD.png")

    # Compute metrics
    avg_accuracy = total_accuracy / (len(dataset) * dataset.max_len) * 100
    avg_loss = total_loss / len(dataset)

    print(f"Validation Total Loss: {avg_loss:.4f}")
    print(f"Validation Action Accuracy: {avg_accuracy:.2f}%")
    print(f"Validation completed in {time.time() - start_time:.2f}s")

    # Log to file
    log_file = "/data/athira/Author_data/New/RGB_revised_data/l_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Validation Total Loss: {avg_loss:.4f}\n")
        f.write(f"Validation Action Accuracy: {avg_accuracy:.2f}%\n")


def validate_policy(model, dataset, device, batch_size=32):
    
    _=val_policy(model,dataset,device,batch_size)
    
def main():

    
    train_trajectory_dir="/data/athira/hm3d_geodesic_trajectory"
    val_trajectory_dir="/data/athira/hm3d_geodesic_trajectory"

    train_dataset = TrajectoryDataset(train_trajectory_dir)
    val_dataset = TrajectoryDataset(val_trajectory_dir,mode="val")
    print(len(train_dataset), len(val_dataset))

    batch_size=16
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.cuda.empty_cache()
    
    model=DepthPolicy(device=device)
    model.to(device)
    model.load_state_dict(torch.load("/data/athira/Author_data/New/depth/rgb_new18.pth", weights_only=True))
    print("dataloading done, ---->")
    train_policy(model, train_dataset, val_dataset,device, batch_size, num_epochs=20)
    # validate_policy(model,val_dataset,device,batch_size=batch_size)



if __name__ == "__main__":
    main()
    print("Training complete!!!")
