import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import yaml

class carla_Dataset(Dataset):
    def __init__(self, data_folder='/share1/pranjal.paul/carla_flow_data/train/'):

        self.data_folder = data_folder
        self.files = sorted([os.path.join(data_folder, f) 
                             for f in os.listdir(data_folder) if f.endswith(".h5")])
        self.learn_angle = False
        self.dataset_index = 0
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)

        self.data_config = all_data_config['carla']
        self.normalize = True
        self.waypoint_spacing = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print('get item',idx)
        with h5py.File(self.files[idx], "r") as f:
            rgb_temporal = torch.from_numpy(f["rgb"][:]).permute(0, 3, 1, 2).float()

            gt_traj = torch.from_numpy(f["gt_traj"][:]).float()  # [32, 2]

            yaw = torch.from_numpy(f["gt_actions"][:3]).float()  # [32, 3]
        
        rgb_temporal = rgb_temporal.reshape(rgb_temporal.size(0)*rgb_temporal.size(1), rgb_temporal.size(2), rgb_temporal.size(3))
        
        goal_point = gt_traj[-1]
        action_mask = True
        if self.learn_angle:
            waypoint = np.concatenate([gt_traj[1:], yaw[1:, None]], axis=-1)
        else:
            waypoint = gt_traj
            
        
        if self.normalize:
            waypoint[:, :2] /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )

        goal_pos = gt_traj[-1]
        # print('goal pos:', goal_pos)
        actions_torch = waypoint  # [31, 2] or [31, 3]
        distance = len(actions_torch)
            
        return (
            torch.as_tensor(rgb_temporal, dtype=torch.float32),
            torch.as_tensor(goal_point, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
