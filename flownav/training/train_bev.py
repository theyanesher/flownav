import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import wandb
from diffusers.training_utils import EMAModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchvision import transforms
from flownav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from flownav.training.logger import Logger
from flownav.training.utils import (
    ACTION_STATS,
    action_reduce,
    compute_losses,
    get_delta,
    normalize_data,
    visualize_action_distribution,
    from_numpy,
)


def train(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger(
        "uc_action_loss", "train", window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", "train", window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        leave=True,
        dynamic_ncols=True,
        colour="magenta",
    ) as tepoch:
        for i, data in enumerate(tepoch):
            # (
            #     obs_image,
            #     goal_image,
            #     actions,
            #     distance,
            #     goal_pos,
            #     _,
            #     action_mask,
            # ) = data
            obs_images = data['context_imgs'] # [1, 6, 3, 224, 400]
            goal_pos = data['future_odom'][:,-1,:] # [1,8,2] -> [1,2]
            actions = data['future_odom'] # [1,8,2]
            # distance = torch.as_tensor(actions.shape[1], dtype=torch.float32) # 8
            distance = data['distance'] # 8
            # action_mask = True
            intrins = data['intrinsics']
            rots = data['context_rots']
            trans = data['context_trans']
            action_mask = (
                (distance < 20)
                and (distance > 3)
            )
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32).to(device)
            # Split the observation image into RGB channels
            # obs_image = obs_image.view(obs_image.size(0), -1,obs_image.size(3), obs_image.size(4)) # (taken care in dataset)
            # breakpoint()
            batch_viz_obs_images = obs_images[:,-1] # the current images in batch
            # )
            # batch_viz_goal_images = TF.resize(
            #     goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
            # )
            batch_viz_goal = goal_pos
            # batch_obs_images = [transform(obs) for obs in obs_images]
            x = obs_images.shape
            batch_obs_images = obs_images.view(x[0], -1, x[3], x[4]) # [1, 18, 224, 400])
            batch_goal = goal_pos.to(device)  
            # Get action mask
            # action_mask = action_mask.to(device)

            # Get naction and normalize it
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            # Get batch size
            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal,
                input_goal_mask=goal_mask,
                intrins = intrins,
                rots = rots,
                trans = trans,
            )

            # Get distance label
            distance = distance.to(device)

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
                1e-2 + (1 - goal_mask.float()).mean()
            )
            # dist_loss = 0
            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Flow
            FM = ConditionalFlowMatcher(sigma=0.0)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)
            vt = model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond
            )

            # L2 loss
            flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

            # Total loss
            loss = flow_loss
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb:
                wandb.log({"total_loss": loss_cpu})
                # wandb.log({"dist_loss": dist_loss.item()})
                wandb.log({"flow_loss": flow_loss.item()})

            if i % print_log_freq == 0:
                losses = compute_losses(
                    ema_model=ema_model.averaged_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal,
                    batch_dist_label=distance.to(device),
                    batch_action_label=actions.to(device),
                    device=device,
                    action_mask=action_mask,
                    intrins = intrins,
                    rots = rots,
                    trans = trans,
                    use_wandb=use_wandb,
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_action_distribution(
                    ema_model=ema_model.averaged_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal,
                    batch_viz_obs_images=batch_viz_obs_images,
                    batch_viz_goal_images=batch_viz_goal,
                    batch_action_label=actions,
                    batch_distance_labels=distance,
                    batch_goal_pos=goal_pos,
                    intrins = intrins,
                    rots = rots,
                    trans = trans,
                    device=device,
                    eval_type="train",
                    project_folder=project_folder,
                    epoch=epoch,
                    num_images_log=num_images_log,
                    num_samples=4,
                    use_wandb=use_wandb,
                )
            # torch.cuda.empty_cache()
