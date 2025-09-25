import habitat
import json
import os
import argparse
import cv2
import torch
import quaternion
import imageio
import shutil
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from policy_agent import Policy_Agent
from data_utils.geometry_tools import *
from config_utils import *
from constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18 as imagenet_resnet18
from matplotlib import pyplot as plt
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def visualize_target(rgb,mask):
    copy_rgb = rgb.copy()
    copy_rgb[mask!=0] = np.array([0,0,255])
    return copy_rgb
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PixelNav_Policy(nn.Module):
    def __init__(self,max_token_length=64,device='cuda:0'):
        super().__init__()
        self.device = device
        self.max_token_length = max_token_length
        # resnet backbone use to encode all the history RGB images, randomly initialized
        self.history_backbone = nn.Sequential(*(list(imagenet_resnet18().children())[:-1]),nn.Flatten()).to(device)
        # goal encoder to encode both the initial RGB image and the goal mask, 4-channel input
        self.goal_backbone = imagenet_resnet18()
        self.goal_backbone.conv1 = nn.Conv2d(4,self.goal_backbone.conv1.out_channels,
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
        # goal concat token shape = (B,1,256), goal input token shape = (B,1,256)

        goal_mask_tensor = torch.as_tensor(goal_mask/255.0,dtype=torch.float32,device=self.device)#.permute(0,3,1,2).contiguous()
        goal_image_tensor = torch.as_tensor(goal_image/255.0,dtype=torch.float32,device=self.device)#.permute(0,3,1,2).contiguous()
        # print(goal_mask_tensor.shape,goal_image_tensor.shape)
        goal_token = self.goal_backbone(torch.concat((goal_image_tensor,goal_mask_tensor),dim=1)).unsqueeze(1)
        goal_concat_token = self.goal_concat_proj(goal_token)
        goal_input_token = self.goal_input_proj(goal_token)  

        # history image token shape = (B,64,512), and the episode input tokens are concated to (B,64,512+256)
        episode_image_tensor = torch.as_tensor(episode_image/255.0,dtype=torch.float32,device=self.device)#.permute(0,1,4,2,3).contiguous()
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
        distance_pred = torch.sigmoid(self.distance_head(out_token))
        # goal_pred = self.goal_head(out_token)
        goal_pred = torch.sigmoid(self.goal_head(out_token)) 
        # print(action_pred.shape,distance_pred.shape,goal_pred.shape)
        return action_pred,distance_pred,goal_pred
class DepthPolicy(nn.Module):

    def __init__(self,max_token_length=64,device='cuda:0'):
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
        # print(self.history_backbone)
        # self.history_backbone.conv1=nn.Conv2d(1,self.history_backbone.conv1.out_channels,
        #                                      kernel_size=self.history_backbone.conv1.kernel_size,
        #                                      stride=self.history_backbone.conv1.stride,
        #                                      padding=self.history_backbone.conv1.padding,
        #                                      bias=self.history_backbone.conv1.bias)
        # self.history_backbone.conv1=conv1
        # goal encoder to encode both the initial RGB image and the goal mask, 4-channel input
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
        # goal concat token shape = (B,1,256), goal input token shape = (B,1,256)
        goal_mask_tensor = torch.as_tensor(goal_mask/255.0,dtype=torch.float32,device=self.device)#.permute(0,3,1,2).contiguous()
        goal_image_tensor = torch.as_tensor(goal_image/255.0,dtype=torch.float32,device=self.device)#.permute(0,3,1,2).contiguous()
        goal_token = self.goal_backbone(torch.concat((goal_image_tensor,goal_mask_tensor),dim=1)).unsqueeze(1)

        goal_concat_token = self.goal_concat_proj(goal_token)
        goal_input_token = self.goal_input_proj(goal_token)  

        # history image token shape = (B,64,512), and the episode input tokens are concated to (B,64,512+256)
        episode_image_tensor = torch.as_tensor(episode_image/255.0,dtype=torch.float32,device=self.device)#.permute(0,1,4,2,3).contiguous()
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
        
        return action_pred 
def random_pixel_goal(habitat_config,habitat_env, difficulty='medium'):
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states['rgb'].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos,robot_rot)
    rgb = camera_obs['rgb']
    depth = camera_obs['depth']
    xs,zs,rgb_points,rgb_colors = get_pointcloud_from_depth(rgb,depth,camera_int)
    rgb_points = translate_to_world(rgb_points,camera_pos,quaternion.as_rotation_matrix(camera_rot))
    if difficulty == 'easy':
        condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & (rgb_points[:,1] > robot_pos[1] - 0.2) & (depth[(zs,xs)][:,0] > 1.0) & (depth[(zs,xs)][:,0] < 3.0))[0]
    elif difficulty == 'medium':
        condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & (rgb_points[:,1] > robot_pos[1] - 0.2) & (depth[(zs,xs)][:,0] > 3.0) & (depth[(zs,xs)][:,0] < 5.0))[0]
    else:
        raise NotImplementedError
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(rgb_points[condition_index])
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_colors[condition_index]/255.0)
    if condition_index.shape[0] == 0:
        return False,[],[],[],[],[],[]
    else:
        random_index = np.random.choice(condition_index)
        target_x = xs[random_index]
        target_z = zs[random_index]
        # print("Target coordinate",target_x,target_z)
        target_point = rgb_points[random_index]
        min_z = max(target_z-5,0)
        max_z = min(target_z+5,depth.shape[0])
        min_x = max(target_x-5,0)
        max_x = min(target_x+5,depth.shape[1])
        target_mask = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)
        target_mask[min_z:max_z,min_x:max_x] = 255
        target_point[1] = robot_pos[1]
        geodesic_distance = habitat_env.sim.geodesic_distance(target_point,robot_pos)
        # cv2.imwrite("goal_mask.jpg",target_mask)
        return True,np.array(rgb),np.array(depth),target_mask,target_point,geodesic_distance,[target_x,target_x]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",type=str,choices=['hm3d','mp3d'],default='mp3d')
    parser.add_argument("--difficulty",type=str,choices=['easy','medium'],default='easy')
    parser.add_argument("--max_length",type=int,default=64)
    parser.add_argument("--episodes",type=int,default=1000)
    parser.add_argument("--robot_height",type=float,default=0.88)
    parser.add_argument("--robot_radius",type=float,default=0.18)
    parser.add_argument("--sensor_height",type=float,default=0.88)
    parser.add_argument("--step_size",type=float,default=0.25)
    parser.add_argument("--turn_angle",type=int,default=30)
    parser.add_argument("--image_width",type=int,default=224)
    parser.add_argument("--image_height",type=int,default=224)
    parser.add_argument("--image_hfov",type=int,default=79)
    args = parser.parse_known_args()[0]
    return args

args = get_args()
if args.prefix == 'mp3d':
    habitat_config = mp3d_data_config(stage='val',
                                 episodes=args.episodes,
                                 robot_height=args.robot_height,
                                 robot_radius=args.robot_radius,
                                 sensor_height=args.sensor_height,
                                 image_width=args.image_width,
                                 image_height=args.image_height,
                                 image_hfov=args.image_hfov,
                                 step_size=args.step_size,
                                 turn_angle=args.turn_angle)
elif args.prefix == 'hm3d':
    habitat_config = hm3d_data_config(stage='val',
                                 episodes=args.episodes,
                                 robot_height=args.robot_height,
                                 robot_radius=args.robot_radius,
                                 sensor_height=args.sensor_height,
                                 image_width=args.image_width,
                                 image_height=args.image_height,
                                 image_hfov=args.image_hfov,
                                 step_size=args.step_size,
                                 turn_angle=args.turn_angle)

env = habitat.Env(habitat_config)
print(POLICY_CHECKPOINT,POLICY_CHECKPOINT_DEPTH)
# policy_agent = Policy_Agent(model_path=POLICY_CHECKPOINT)
# model=policy_agent.network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_d=DepthPolicy().to(device)
print("Depth Policy Model : ")
print("Total parameters : ",sum(p.numel() for p in model_d.parameters()))
print("Trainable p/m : ",sum(p.numel() for p in model_d.parameters() if p.requires_grad))
model=PixelNav_Policy(device=device)
model.load_state_dict(torch.load(POLICY_CHECKPOINT, map_location="cuda:0"))
model.eval()
if os.path.exists(POLICY_CHECKPOINT_DEPTH)==True:
    model_d.load_state_dict(torch.load(POLICY_CHECKPOINT_DEPTH, weights_only=True))
print("PixelNav Policy Model : ")
print("Total parameters : ",sum(p.numel() for p in model.parameters()))
print("Trainable p/m : ",sum(p.numel() for p in model.parameters() if p.requires_grad))
model_d.eval()
# exit()
oracle_agent = ShortestPathFollower(env.sim,0.5,False)
metrics_sr = {'easy':[],'medium':[],'hard':[]}
metrics_spl = {'easy':[],'medium':[],'hard':[]}
valid_actions=['stop', 'move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down']
for i in tqdm(range(args.episodes)):
    # os.makedirs("%s_eval_trajectory/evaluate_%s_%d/"%(args.prefix,args.prefix,i),exist_ok=True)
    timesteps = 0
    obs = env.reset()
    goal_flag,goal_image,goal_depth,goal_mask,goal_point,goal_dist,goal_coord = random_pixel_goal(habitat_config,env,args.difficulty)
    if goal_flag == False or oracle_agent.get_next_action(goal_point) == 0:
        # shutil.rmtree("%s_eval_trajectory/evaluate_%s_%d/"%(args.prefix,args.prefix,i))
        continue
    max_token_length=64
    history_image = np.zeros((max_token_length,224,224,3))
    history_depth = np.zeros((max_token_length,224,224,1))
    # goal_image = cv2.resize(cv2.cvtColor(goal_image,cv2.COLOR_BGR2RGB),(160,120))
    goal_image = goal_image[np.newaxis,:,:,:]
    # goal_mask = cv2.resize(goal_mask,(160,120),cv2.INTER_NEAREST)
    goal_mask = goal_mask[np.newaxis,:,:,np.newaxis]
    goal_depth = goal_depth[np.newaxis,:,:,:]
    # print("After reshaping---------",goal_image.shape,goal_mask.shape)
    collide_times = 0
    collide_action = 0
    
    move_distance = 0
    last_position = env.sim.get_agent_state().position
    while True:
        if env.episode_over or timesteps >= args.max_length:
            # print(env.episode_over,timesteps)
            timesteps=0

            break          
        
        image=obs['rgb']
        depth=obs['depth']
        collide = env.sim.previous_step_collided
        # current_obs=image

        # print(current_obs.shape,history_image.shape)
        history_image[timesteps] = image.copy() #append(self.current_obs)
        history_depth[timesteps] = depth.copy() #append(self.current_obs)
        input_image = history_image[np.newaxis,:,:,:,:]
        input_depth = history_depth[np.newaxis,:,:,:,:]
        goal_mask_tensor = torch.from_numpy(goal_mask).permute(0, 3, 1, 2).float().to(device)
        goal_depth_tensor = torch.from_numpy(goal_depth).permute(0, 3, 1, 2).float().to(device)
        goal_image_tensor = torch.from_numpy(goal_image).permute(0, 3, 1, 2).float().to(device)
        input_image_tensor = torch.from_numpy(input_image).permute(0, 1, 4, 2, 3).float().to(device)
        input_depth_tensor=torch.from_numpy(input_depth).permute(0, 1, 4, 2, 3).float().to(device)

        # Now pass to the model
        action_pred_1,_,_ = model(goal_mask_tensor, goal_image_tensor, input_image_tensor)
        action_pred_2 = model_d(goal_mask_tensor, goal_depth_tensor, input_depth_tensor)
        pred_1=action_pred_1[0][timesteps]
        pred_2=action_pred_2[0][timesteps]
        if torch.isnan(pred_1).any():
            predicted_action=pred_1.detach().cpu().numpy().argmax()
        elif torch.isnan(pred_2).any():
            predicted_action=pred_2.detach().cpu().numpy().argmax()
        else:
            # avg_logits= (F.log_softmax(pred_1, dim=-1) + F.log_softmax(pred_2, dim=-1)) / 2
            # predicted_action = torch.argmax(avg_logits, dim=-1)

            predicted_action= ((9* pred_1.detach().cpu().numpy() + pred_2.detach().cpu().numpy()) / 10).argmax()

        obs=env.step({"action":predicted_action})
        # obs = env.step(predicted_actions.detach().cpu())
        move_distance += np.sqrt(np.sum(np.square(last_position - env.sim.get_agent_state().position)))
        last_position = env.sim.get_agent_state().position
        timesteps += 1
        
        
    if goal_dist < 3.0:
        sr = (np.sqrt(np.sum(np.square(goal_point - env.sim.get_agent_state().position))) < 1.0)
        spl = np.clip(sr * goal_dist / (move_distance+1e-6),0,1)
        metrics_sr['easy'].append(sr)
        metrics_spl['easy'].append(spl)
    elif goal_dist > 3.0 and goal_dist < 5.0:
        sr = (np.sqrt(np.sum(np.square(goal_point - env.sim.get_agent_state().position))) < 1.0)
        spl = np.clip(sr * goal_dist / (move_distance+1e-6),0,1)
        metrics_sr['medium'].append(sr)
        metrics_spl['medium'].append(spl)
    print("HM3D",POLICY_CHECKPOINT)

    print('easy episode = %d'%len(metrics_sr['easy']))
    print('easy sr = %.4f'%np.array(metrics_sr['easy']).mean())
    print('easy spl = %.4f'%np.array(metrics_spl['easy']).mean())

    print('medium episode = %d'%len(metrics_sr['medium']))
    print('medium sr = %.4f'%np.array(metrics_sr['medium']).mean())
    print('medium spl = %.4f'%np.array(metrics_spl['medium']).mean())
    # image_writer.close()


       
    