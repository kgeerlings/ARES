from tqdm import tqdm
from collections import defaultdict
import os
import torch
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


CHECKPOINT_PATH = "models/3_ally_semi_dodges_enemies.pt"
TRAINING_NAME = "3.2_ally_dodges_enemies_finetune"


def load_checkpoint_to_train(
    checkpoint_path, policy, critic=None, optimizer=None, loss_module=None
):
    """
    Load a saved checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        policy (torch.nn.Module): The policy model to load the state into.
        critic (torch.nn.Module, optional): The critic model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        loss_module (torch.nn.Module, optional): The loss module to load the state into.

    Returns:
        dict: Loaded checkpoint data.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load policy state
    policy.load_state_dict(checkpoint["policy_state_dict"])
    print(f"Policy loaded from iteration {checkpoint['iteration']}")

    # Load critic state if provided
    if critic is not None and "critic_state_dict" in checkpoint:
        critic.load_state_dict(checkpoint["critic_state_dict"])
        print("Critic state loaded")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded")

    # Load loss module state if provided
    if loss_module is not None and "loss_module_state_dict" in checkpoint:
        loss_module.load_state_dict(checkpoint["loss_module_state_dict"])
        print("Loss module state loaded")

    print(f"Mean reward from checkpoint: {checkpoint['reward_mean']:.4f}")
    return checkpoint


# Load checkpoint before training loop
checkpoint_path = CHECKPOINT_PATH
checkpoint = load_checkpoint_to_train(checkpoint_path, policy, critic=critic)

# Get starting iteration from checkpoint
start_iteration = 0
if checkpoint is not None:
    start_iteration = checkpoint["iteration"] + 1
    print(f"Resuming training from iteration {start_iteration}")
else:
    print("Starting training from scratch")


# Callbacks
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"runs/PPO_finetuning_{TRAINING_NAME}"
writer = SummaryWriter(log_dir)
logs = defaultdict(list)

# Training loop ======================
tqdm.format_sizeof = lambda x, divisor=None: f"{x:,}" if divisor else f"{x:5.2f}"
progress_bar = tqdm(total=collector.total_frames, desc=None, unit_scale=True)

for iteration, batch in enumerate(collector, start=start_iteration):

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

            # Before the backward pass, check if total_loss is NaN
            if torch.isnan(total_loss).any():
                print(
                    f"NaN detected in total_loss at iteration {iteration}. Skipping step."
                )
                # skipping the step
                continue

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

    # Count episodes where ally reached target (transitions from 0 to 1)
    # Assuming the observation contains ally_reached_target at index 2
    current_ally_reached = batch["next", "observation"][:, 2]
    prev_ally_reached = batch["observation"][:, 2]
    # Count when it transitions from 0 (or False) to 1 (True)
    ally_reached_target_count = ((current_ally_reached == 1.0) & (prev_ally_reached == 0.0)).sum().item()

    # Calculate explained variance
    with torch.no_grad():
        values = batch["state_value"].flatten()
        returns = batch["value_target"].flatten()
        var_returns = returns.var()
        explained_var = 1 - (returns - values).var() / (var_returns + 1e-8)
        explained_var = explained_var.item()

    logs["reward"].append(reward_mean)
    logs["lr"].append(current_lr)
    logs["done_count"].append(done_count)
    logs["terminated_count"].append(terminated_count)
    logs["truncated_count"].append(truncated_count)
    logs["ally_reached_target_count"].append(ally_reached_target_count)
    logs["explained_variance"].append(explained_var)

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
    writer.add_scalar(
        "Training/Ally_Reached_Target_Count", ally_reached_target_count, iteration
    )  # Number of times ally reached target
    writer.add_scalar(
        "Training/Explained_Variance", explained_var, iteration
    )  # Explained variance of value function

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
