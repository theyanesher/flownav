import h5py
import os

path = '/share1/pranjal.paul/carla_flow_data/train/'
files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")])

def get_data(i, files=files):
    return h5py.File(files[i], "r")

data = get_data(0)
breakpoint() 


"""
    - rgb_temporal = f["rgb"][:]  # [5, 600, 800, 3]

    - depth_temporal = f["depth"][:]  # [5, 600, 800, 3]

    - obstacle_pose_ego = f["obstacle_pose"][:]  # [10, 8]: [x, y, yaw, length_extent, width_extent]

    - ego_state = f["ego_state"][:]  # [8,]: [x, y, yaw, vel_x, vel_y, accl_x, acc_y, yaw_rate]. (NOTE: Here x, y are in global coordinate). For flownav, (x, y, yaw) will be 0

    - gt_traj_ego = f["gt_traj"][:]  # [32, 2]

    - gt_actions = f["gt_actions"][:]  # [32, 3]: [accl_x, accl_y, steering]. NOTE: For flownav, if they denoise actions, then you will need (accl_x, steering) pair only

    - reference_path_ego = f["reference_path_ego"][:]  # [1000, 3] # [x, y, yaw]

    In gt_traj, gt_actions:
    planning horizon is 6.4 seconds with dt=0.2 i.e. 32 timsteps, hence 32.
"""