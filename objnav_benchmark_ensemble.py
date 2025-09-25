import habitat
import os
import argparse
import csv
import cv2
import imageio
import numpy as np
from cv_utils.detection_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config,mp3d_config
# from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent_Ensemble as Policy_Agent
from cv_utils.detection_tools import initialize_dino_model
from cv_utils.segmentation_tools import initialize_sam_model
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
import numpy as np
# from llm_utils.gpt_request import gptv_response
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
import cv2
import ast
import json
import openai
import base64

# import openai
# import base64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-base", torch_dtype=torch.float16).to(device)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

class BLIP_Planner:
    def __init__(self,dino_model,sam_model):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
    
    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
            # self.object_goal = 'chair'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image
    def llava_direction_reasoning(self,panorama_imgs, goal_category):
        directions = ["front right", "back right", "back","back left", "front left","front"]
        best_score = -float('inf')
        best_idx = 0
        for idx, (img, dir_label) in enumerate(zip(panorama_imgs, directions)):
            prompt = (
                f"This is the {dir_label} view. "
                f"Does this direction look promising to find a {goal_category}? "
                "Answer yes or no and explain briefly."
            )
            img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)  # Normalize to 0–1
            img_uint8 = (img_norm * 255).astype('uint8')            # Scale to 0–255 and convert
            print(img_uint8.shape)
            img_gray = img_uint8.mean(axis=-1).astype('uint8')
            img_pil = Image.fromarray(img_gray, mode='L')            # img_pil = Image.fromarray(img)
            print(img_pil.size)
            inputs = processor(img_pil, prompt, return_tensors="pt").to(device, torch.float16)


            # inputs = llava_processor( img_pil, prompt,return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs)
                answer=processor.decode(out[0], skip_special_tokens=True).lower()
                # output = llava_model.generate(**inputs, max_new_tokens=64)
                # answer = llava_processor.decode(output[0], skip_special_tokens=True).lower()
            score = 1 if "yes" in answer else -1
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx
    
    def make_plan(self,pano_images,pano_depths=[]):
        direction= self.llava_direction_reasoning(pano_images, self.object_goal)
        goal_depth = pano_depths[direction]
        # direction,goal_flag = self.query_gpt4v(pano_images)
        goal_flag=True
        direction_image = pano_images[direction]
        goal_image=direction_image.copy()
        if direction_image.shape[2] != 3:
            depth_norm = ((direction_image - direction_image.min()) / (direction_image.ptp() + 1e-8) * 255).astype('uint8')
            depth_norm = np.stack([depth_norm]*3, axis=-1)  # (H, W, 3)
            direction_image = depth_norm
        
        target_bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),self.detect_objects,self.dino_model)
        if self.object_goal not in self.detect_objects:
            goal_flag = False
        elif self.detect_objects.index(self.object_goal) not in target_bbox.class_id:
            goal_flag = False

        if goal_flag:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),[self.object_goal],self.dino_model)    
        else:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),['floor'],self.dino_model)
        try:
            mask = sam_masking(direction_image,bbox.xyxy,self.sam_model)
        except:
            mask = np.ones_like(direction_image).mean(axis=-1)
        
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(mask)

        debug_image = np.array(direction_image)
        debug_mask = np.zeros_like(debug_image)
        pixel_y,pixel_x = np.where(mask>0)[0:2]
        pixel_y = int(pixel_y.mean())
        pixel_x = int(pixel_x.mean())
        debug_image = cv2.rectangle(debug_image,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,0,0),-1)
        debug_mask = cv2.rectangle(debug_mask,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,255,255),-1)
        debug_mask = debug_mask.mean(axis=-1)
        return goal_image,debug_mask,goal_depth,debug_image,direction,goal_flag
        

def write_metrics(metrics,path="/data/athira/Author_data/Ensemble_BLIP_mp3d_75.csv"):
    with open(path, mode="a", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=100)
    return parser.parse_known_args()[0]

def detect_mask(image,category,detect_model):
    det_result = openset_detection(image,category,detect_model)
    if det_result.xyxy.shape[0] > 0:
        goal_image = image
        goal_mask_xyxy = det_result.xyxy[np.argmax(det_result.confidence)]
        goal_mask_x = int((goal_mask_xyxy[0]+goal_mask_xyxy[2])/2)
        goal_mask_y = int((goal_mask_xyxy[1]+goal_mask_xyxy[3])/2)
        goal_mask = np.zeros((goal_image.shape[0],goal_image.shape[1]),np.uint8)
        goal_mask = cv2.rectangle(goal_mask,(goal_mask_x-8,goal_mask_y-8),(goal_mask_x+8,goal_mask_y+8),(255,255,255),-1)
        return True,goal_image,goal_mask
    return False,[],[]


args = get_args()
habitat_config = mp3d_config(stage='val',episodes=args.eval_episodes)

# habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
habitat_env = habitat.Env(habitat_config)
detection_model = initialize_dino_model()
segmentation_model = initialize_sam_model()

nav_planner = BLIP_Planner(detection_model,segmentation_model)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT,depth_model_path=POLICY_CHECKPOINT_DEPTH)
evaluation_metrics = []
os.makedirs("/data/athira/Author_data/Ensemble_BLIP_mp3d_75",exist_ok=True)
for i in tqdm(range(args.eval_episodes)):
    if True:
        find_goal = False
        obs = habitat_env.reset()
        dir = "/data/athira/Author_data/Ensemble_BLIP_mp3d_75/trajectory_"+str(i)+str(habitat_env.current_episode.object_category)

        # dir = "/data/athira/Author_data/Depth_BLIP/trajectory_%d"%i
        os.makedirs(dir,exist_ok=True)
        fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
        topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
        heading_offset = 0

        nav_planner.reset(habitat_env.current_episode.object_category)
        episode_images = [obs['rgb']]
        episode_depths = [obs['depth']]
        episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

        # a whole round planning process
        for _ in range(11):
            obs = habitat_env.step(3)
            episode_images.append(obs['rgb'])
            episode_depths.append(obs['depth'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        goal_image,goal_mask,goal_depth,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(episode_images[-12:],episode_depths[-12:])    
        for j in range(min(11-goal_rotate,1+goal_rotate)):
            if goal_rotate <= 6:
                obs = habitat_env.step(3)
                episode_images.append(obs['rgb'])
                episode_depths.append(obs['depth'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            else:
                obs = habitat_env.step(2)
                episode_images.append(obs['rgb'])
                episode_depths.append(obs['depth'])

                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        nav_executor.reset(goal_image,goal_mask,goal_depth)


        while not habitat_env.episode_over:
            action,skill_image = nav_executor.step(obs['rgb'],obs['depth'],habitat_env.sim.previous_step_collided)
            if action != 0 or goal_flag:
                if action == 4:
                    heading_offset += 1
                elif action == 5:
                    heading_offset -= 1
                # action = {"action": action}  # Convert tensor to dict

                # obs = habitat_env.step(action)
                obs = habitat_env.step(action)
                episode_images.append(obs['rgb'])
                episode_depths.append(obs['depth'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            else:
                if habitat_env.episode_over:
                    break
                
                for _ in range(0,abs(heading_offset)):
                    if habitat_env.episode_over:
                        break
                    if heading_offset > 0:
                        obs = habitat_env.step(5)
                        episode_images.append(obs['rgb'])
                        episode_depths.append(obs['depth'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        heading_offset -= 1
                    elif heading_offset < 0:
                        obs = habitat_env.step(4)
                        episode_images.append(obs['rgb'])
                        episode_depths.append(obs['depth'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        heading_offset += 1
                if habitat_env.episode_over:
                        break
                obs=habitat_env.step(1)
                # a whole round planning process
                for _ in range(11):
                    if habitat_env.episode_over:
                        break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_depths.append(obs['depth'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                goal_image,goal_mask,goal_depth,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(episode_images[-12:],episode_depths[-12:])
                for j in range(min(11-goal_rotate,goal_rotate+1)):
                    if habitat_env.episode_over:
                        break
                    if goal_rotate <= 6:
                        obs = habitat_env.step(3)
                        episode_images.append(obs['rgb'])
                        episode_depths.append(obs['depth'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    else:
                        obs = habitat_env.step(2)
                        episode_images.append(obs['rgb'])
                        episode_depths.append(obs['depth'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                nav_executor.reset(goal_image,goal_mask,goal_depth)
        
        for image,topdown in zip(episode_images,episode_topdowns):
            fps_writer.append_data(image)
            topdown_writer.append_data(topdown)
        fps_writer.close()
        topdown_writer.close()

        evaluation_metrics.append({'success':habitat_env.get_metrics()['success'],
                                'spl':habitat_env.get_metrics()['spl'],
                                'sspl':habitat_env.get_metrics()['soft_spl'],
                                'distance_to_goal':habitat_env.get_metrics()['distance_to_goal'],
                                'object_goal':habitat_env.current_episode.object_category})
        write_metrics(evaluation_metrics)
            

        

            

        

        


    
        

