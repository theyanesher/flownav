import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
import numpy as np
import os


class NuScenesTemporalDataset(Dataset):
    def __init__(self,
                 nusc_root,
                 version='v1.0-trainval',
                 camera='CAM_FRONT',
                 context_len=5,
                 future_len=8,
                 transform=None,
                 include_goal_image=True,
                 resize_hw=(224, 400)):
        """
        Returns for each valid timestep:
          - context + current images (total context_len+1)
          - their intrinsics, rotations, and translations
          - future `future_len` odometry steps
          - (optional) goal image and its intrinsics
        """
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        self.camera = camera
        self.context_len = context_len
        self.future_len = future_len
        self.transform = transform
        self.include_goal_image = include_goal_image
        self.resize_hw = resize_hw  # (H, W)

        self.scene_sample_tokens = []
        self.valid_indices = []

        # --- Collect samples from all scenes ---
        for scene in self.nusc.scene:
            first_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_token)
            scene_samples = []

            while True:
                scene_samples.append(sample['token'])
                if sample['next'] == '':
                    break
                sample = self.nusc.get('sample', sample['next'])

            # Only consider long enough trajectories
            if len(scene_samples) > self.context_len + self.future_len:
                self.scene_sample_tokens.append(scene_samples)

        # --- Create valid (scene_idx, local_idx) pairs ---
        for s_idx, scene_tokens in enumerate(self.scene_sample_tokens):
            for i in range(self.context_len, len(scene_tokens) - self.future_len - 1):
                self.valid_indices.append((s_idx, i))

    def __len__(self):
        return len(self.valid_indices)

    def _get_image_intrin_extrin(self, sample_token):
        """
        Loads image, intrinsics, rotation, translation for given sample.
        Applies resize adjustment to intrinsics.
        """
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][self.camera])
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # --- Camera calibration (extrinsics + intrinsics) ---
        calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsics = np.array(calib['camera_intrinsic'], dtype=np.float32)
        rot = torch.tensor(Quaternion(calib['rotation']).rotation_matrix, dtype=torch.float32)  # [3,3]
        trans = torch.tensor(calib['translation'], dtype=torch.float32)  # [3]

        # --- Resize intrinsics based on image transform ---
        new_h, new_w = self.resize_hw
        sx = new_w / orig_w
        sy = new_h / orig_h
        intrinsics[0, 0] *= sx  # fx
        intrinsics[0, 2] *= sx  # cx
        intrinsics[1, 1] *= sy  # fy
        intrinsics[1, 2] *= sy  # cy

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(intrinsics, dtype=torch.float32), rot, trans

    def _get_odom(self, sample_token):
        """Extract (x, y, yaw) from ego pose."""
        sample = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        x, y, z = pose['translation']
        q = Quaternion(pose['rotation'])
        yaw = q.yaw_pitch_roll[0]
        return np.array([x, y], dtype=np.float32)

    def __getitem__(self, idx):
        scene_idx, local_idx = self.valid_indices[idx]
        scene_tokens = self.scene_sample_tokens[scene_idx]

        # --- Context + Current images ---
        context_imgs, context_intrins, context_rots, context_trans = [], [], [], []
        for j in range(local_idx - self.context_len, local_idx + 1):  # include current
            token = scene_tokens[j]
            img, intrin, rot, trans = self._get_image_intrin_extrin(token)
            context_imgs.append(img)
            context_intrins.append(intrin)
            context_rots.append(rot)
            context_trans.append(trans)

        context_imgs = torch.stack(context_imgs, dim=0)         # [context_len+1, 3, H, W]
        context_intrins = torch.stack(context_intrins, dim=0)   # [context_len+1, 3, 3]
        context_rots = torch.stack(context_rots, dim=0)         # [context_len+1, 3, 3]
        context_trans = torch.stack(context_trans, dim=0)       # [context_len+1, 3]

        # --- The last one in context is current ---
        current_img = context_imgs[-1]
        current_intrin = context_intrins[-1]
        current_rot = context_rots[-1]
        current_trans = context_trans[-1]

        # --- Future odometry ---
        future_tokens = scene_tokens[local_idx : local_idx + self.future_len]
        future_odom = [self._get_odom(t) for t in future_tokens]
        future_odom = torch.tensor(np.stack(future_odom), dtype=torch.float32)
        relative_odom = future_odom - future_odom[0]

        # --- Goal image ---
        goal_img, goal_intrin, goal_rot, goal_trans = None, None, None, None
        if self.include_goal_image and (local_idx + self.future_len < len(scene_tokens)):
            goal_img, goal_intrin, goal_rot, goal_trans = self._get_image_intrin_extrin(
                scene_tokens[local_idx + self.future_len]
            )

        # --- Repeat intrinsics 6× for uniform shape ---
        repeated_intrin = torch.stack([current_intrin] * (self.context_len + 1), dim=0)

        return {
            "current_img": current_img,            # [3, H, W]
            "context_imgs": torch.as_tensor(context_imgs, dtype=torch.float32),          # [context_len+1, 3, H, W]
            "future_odom": torch.as_tensor(relative_odom, dtype=torch.float32),          # [future_len, 3]
            "goal_img": goal_img,                  # [3, H, W] or None
            "intrinsics": torch.as_tensor(repeated_intrin, dtype=torch.float32),         # [6, 3, 3]
            "context_intrins": context_intrins,    # [6, 3, 3]
            "context_rots": torch.as_tensor(context_rots, dtype=torch.float32),          # [6, 3, 3]
            "context_trans": torch.as_tensor(context_trans, dtype=torch.float32),        # [6, 3]
            "distance": torch.as_tensor(relative_odom.shape[0], dtype=torch.float32),
            "scene_index": scene_idx,
            "sample_index": local_idx
        }
