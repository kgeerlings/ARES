from tqdm import tqdm
from collections import defaultdict
import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

from ares.torchrl_setup.collector import collector
from ares.torchrl_setup.replay_buffer import buffer
from ares.torchrl_setup.loss import loss_module, optimizer
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.critic import critic
from ares.torchrl_setup.hyperparameters_and_setup import (
    frames_per_batch,
    minibatch_size,
)
from ares.torchrl_setup.hyperparameters_and_setup import (
    clip_epsilon,
    entropy_coef,
    critic_coef,
    gamma,
    lmbda,
    max_grad_norm,
    learning_rate,
    num_epochs,
)


# Callbacks
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"runs/ares_runs_{current_time}"
writer = SummaryWriter(log_dir)
logs = defaultdict(list)

# Training loop ======================
tqdm.format_sizeof = lambda x, divisor=None: f"{x:,}" if divisor else f"{x:5.2f}"
progress_bar = tqdm(total=collector.total_frames, desc=None, unit_scale=True)

for iteration, batch in enumerate(collector):

    batch = loss_module.value_estimator(
        batch,
    )
    batch = batch.reshape(-1)
    buffer.extend(batch)

    epoch_losses = defaultdict(list)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):

            sample = buffer.sample()

            loss_vals = loss_module(sample)
            total_loss = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            epoch_losses["objective"].append(loss_vals["loss_objective"].item())
            epoch_losses["critic"].append(loss_vals["loss_critic"].item())
            epoch_losses["entropy"].append(loss_vals["loss_entropy"].item())
            epoch_losses["total"].append(total_loss.item())

            # Get optimizer relative to the loss
            optimizer.zero_grad()

            # Backward propagation
            total_loss.backward()

            # Clip gradient
            params = optimizer.param_groups[0]["params"]
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

            # Push a step to the optimizer
            optimizer.step()

    if hasattr(policy, "step"):
        policy.step(batch.numel())

    collector.update_policy_weights_()
    progress_bar.update(batch.numel())

    # Log losses
    for key, val_list in epoch_losses.items():
        writer.add_scalar(
            f"Training/Loss_{key.capitalize()}",
            torch.tensor(val_list).mean().item(),
            iteration,
        )

    # Training logs
    reward_mean = batch["next", "reward"].mean().item()
    reward_std = batch["next", "reward"].std().item()
    current_lr = optimizer.param_groups[0]["lr"]

    # Count done, terminated, and truncated episodes
    done_count = batch["next", "done"].sum().item()
    terminated_count = batch["next", "terminated"].sum().item()
    truncated_count = batch["next", "truncated"].sum().item()

    logs["reward"].append(reward_mean)
    logs["lr"].append(current_lr)
    logs["done_count"].append(done_count)
    logs["terminated_count"].append(terminated_count)
    logs["truncated_count"].append(truncated_count)

    # Logging TensorBoard
    writer.add_scalar("Training/Reward_Mean", reward_mean, iteration)
    writer.add_scalar("Training/Reward_Std", reward_std, iteration)
    writer.add_scalar("Training/LearningRate", current_lr, iteration)
    writer.add_scalar("Training/Done_Count", done_count, iteration)
    writer.add_scalar(
        "Training/Terminated_Count", terminated_count, iteration
    )  # Number of failures (cartpole falls)
    writer.add_scalar(
        "Training/Truncated_Count", truncated_count, iteration
    )  # Number of successes

    # Record env for evaluation
    # if iteration % 10 == 0:
    #     env_to_video(iteration)

    # Save MODEL
    if iteration % 100 == 0 and iteration > 0:
        checkpoint = {
            "iteration": iteration,
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_module_state_dict": loss_module.state_dict(),
            "reward_mean": reward_mean,
            "logs": logs,
            "hyperparameters": {
                "clip_epsilon": clip_epsilon,
                "entropy_coef": entropy_coef,
                "critic_coef": critic_coef,
                "gamma": gamma,
                "lmbda": lmbda,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "frames_per_batch": frames_per_batch,
                "minibatch_size": minibatch_size,
            },
        }
        save_dir = "checkpoints"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"checkpoint_iter_{iteration}.pt")
        torch.save(checkpoint, save_path)
        print(f"Modèle sauvegardé: {save_path}")

        # Optionnel: garder seulement les 5 derniers checkpoints
        checkpoint_files = sorted(
            [f for f in os.listdir(save_dir) if f.startswith("checkpoint_iter_")]
        )
