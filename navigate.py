import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from flownav.data.vint_dataset import ViNT_Dataset
from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

import matplotlib.pyplot as plt
import yaml
import torchdiffeq

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped, PoseStamped, Point

from deployment.utils import msg_to_pil, to_numpy, transform_images, load_model

from flownav.training.utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
# from topic_names import (IMAGE_TOPIC,
#                         WAYPOINT_TOPIC,
#                         SAMPLED_ACTIONS_TOPIC)

IMAGE_TOPIC = '/realsense/color/image_raw'
WAYPOINT_TOPIC = '/waypoint'
SAMPLED_ACTIONS_TOPIC = '/sampled_waypoint'

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/theyanesh.er/flownav/deployment/"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="robot.yaml"
MODEL_CONFIG_PATH = "/home/theyanesh.er/flownav/deployment/models.yaml"
# with open(ROBOT_CONFIG_PATH, "r") as f:
#     robot_config = yaml.safe_load(f)
MAX_V = 0.2
MAX_W = 0.4
RATE = 4

# GLOBALS
context_queue = []
context_size = None  
subgoal = []
trajs_pub = rospy.Publisher('/trajectories', MarkerArray, queue_size=1)

def load_model(model_path, config, device=torch.device("cuda")):
    # Create the model

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"],
    )
    print("Loaded NoMaD_ViNT vision encoder",config["encoding_size"])
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    # print('model keys',model.state_dict().keys())

    checkpoint = torch.load(
    config["depth"]["weights_path"],
    map_location=device,
    )
    saved_state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    updated_state_dict = {
        k.replace("pretrained.", ""): v
        for k, v in saved_state_dict.items()
        if "pretrained" in k
    }
    new_state_dict = {
        k: v
        for k, v in updated_state_dict.items()
        if k in model.vision_encoder.depth_encoder.state_dict()
    }
    model.vision_encoder.depth_encoder.load_state_dict(new_state_dict, strict=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Some checkpoints wrap the state dict inside "model"
    if "model" in checkpoint:
        ckpt_state = checkpoint["model"]
    else:
        ckpt_state = checkpoint

    print("\n--- Model keys ---")
    print(list(model.state_dict().keys())[:20], "...\n")

    print("--- Checkpoint keys ---")
    print(list(ckpt_state.keys())[:20], "...\n")

    # Try loading with strict=False to see mismatches
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print(">> Missing keys in checkpoint:", missing)
    print(">> Unexpected keys in checkpoint:", unexpected)



    latest_checkpoint = torch.load(model_path, map_location=device)
    # print('latest check',latest_checkpoint.keys())
    if "model" in latest_checkpoint:
        model.load_state_dict(latest_checkpoint["model"], strict=True)
    else:
        model.load_state_dict(latest_checkpoint, strict=True)
    
    model = model.to(device)
    model.eval()
    return model

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def publish_trajectories(trajs, sdf_id, non_sdf_id):
    marker_array = MarkerArray()
    for i, trajectory in enumerate(trajs):  # list_of_trajectories: list of Nx3 arrays
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = i
        marker.scale.x = 0.05  # line width
        # Set marker color based on trajectory index
        if i == sdf_id:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # Green
        elif i == non_sdf_id:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0  # Blue
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0  # Red
        marker.color.a = 1.0
        marker.points = [Point(x, y, 0) for x, y in trajectory]
        marker_array.markers.append(marker)

    trajs_pub.publish(marker_array)


def main(args: argparse.Namespace):
    global context_size

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()
    
     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    # if model_params["model_type"] == "nomad":
    #     num_diffusion_iters = model_params["num_diffusion_iters"]
        # noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=model_params["num_diffusion_iters"],
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )
    # navigation loop
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action
                    traj = torchdiffeq.odeint(
                        lambda t, x: model.forward(
                            "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
                        ),
                        noisy_action,
                        torch.linspace(0, 1, 10, device=device),
                        atol=1e-4,
                        rtol=1e-4,
                        method="euler",
                    )
                    # init scheduler
                    # noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    # for k in noise_scheduler.timesteps[:]:
                    #     # predict noise
                    #     noise_pred = model(
                    #         'noise_pred_net',
                    #         sample=naction,
                    #         timestep=k,
                    #         global_cond=obs_cond
                    #     )
                    #     # inverse diffusion step (remove noise)
                    #     naction = noise_scheduler.step(
                    #         model_output=noise_pred,
                    #         timestep=k,
                    #         sample=naction
                    #     ).prev_sample
                    # t = 0.0
                    # dt = 1.0 / float(num_diffusion_iters)
                    # for step in range(num_diffusion_iters):
                    #     v_t = model(
                    #         "noise_pred_net",      
                    #         sample=naction,
                    #         timestep=t,
                    #         global_cond=obs_cond
                    #     )
                    #     naction = naction + dt * v_t
                    #     t += dt

                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(traj[-1]))
                publish_trajectories(naction, 2, 4)
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
            else:
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="husky",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)

