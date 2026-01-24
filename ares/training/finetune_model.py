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

class FineTuner:
    """Class to handle the fine-tuning of the RL agent."""

    def __init__(
            self, 
            collector=collector, 
            buffer=buffer, 
            loss_module=loss_module, 
            optimizer=optimizer, 
            policy=policy, 
            critic=critic, 
            frames_per_batch=frames_per_batch, 
            minibatch_size=minibatch_size, 
            clip_epsilon=clip_epsilon, 
            entropy_coef=entropy_coef, 
            critic_coef=critic_coef, 
            gamma=gamma, 
            lmbda=lmbda, 
            max_grad_norm=max_grad_norm, 
            learning_rate=learning_rate, 
            num_epochs=num_epochs
            ) -> None:
        """Initialize the FineTuner class."""

        # TorchRL setup components
        self.collector = collector
        self.buffer = buffer
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.policy = policy
        self.critic = critic

        # Hyperparameters
        self.frames_per_batch = frames_per_batch
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.gamma = gamma
        self.lmbda = lmbda
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Training parameters
        self.checkpoint_path = "models/3_ally_semi_dodges_enemies.pt"
        self.training_name = "3.2_ally_dodges_enemies_finetune"

        # Callbacks
        self.writer = SummaryWriter(f"runs/finetuning_{self.training_name}")
        self.logs = defaultdict(list)


    def _load_checkpoint_to_train(self) -> dict:
        """
        Load a saved checkpoint.

        Returns:
            dict: Loaded checkpoint data.
        """
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return None

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        # Load policy state
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Policy loaded from iteration {checkpoint['iteration']}")

        # Load critic state if provided
        if self.critic is not None and "critic_state_dict" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            print("Critic state loaded")

        # Load optimizer state if provided
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded")

        # Load loss module state if provided
        if self.loss_module is not None and "loss_module_state_dict" in checkpoint:
            self.loss_module.load_state_dict(checkpoint["loss_module_state_dict"])
            print("Loss module state loaded")

        print(f"Mean reward from checkpoint: {checkpoint['reward_mean']:.4f}")
        return checkpoint


    def train(self) -> None:
        """
        Fine-tuning loop for the RL agent.
        
        Returns:
            None
        """

        # Load checkpoint before training loop
        checkpoint = self._load_checkpoint_to_train()

        # Get starting iteration from checkpoint
        start_iteration = 0
        if checkpoint:
            start_iteration = checkpoint["iteration"] + 1
            print(f"Resuming training from iteration {start_iteration}")
        else:
            print("Starting training from scratch")

        # Training loop
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
                self.writer.add_scalar(
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

            self.logs["reward"].append(reward_mean)
            self.logs["lr"].append(current_lr)
            self.logs["done_count"].append(done_count)
            self.logs["terminated_count"].append(terminated_count)
            self.logs["truncated_count"].append(truncated_count)
            self.logs["ally_reached_target_count"].append(ally_reached_target_count)
            self.logs["explained_variance"].append(explained_var)

            # Logging TensorBoard
            self.writer.add_scalar("Training/Reward_Mean", reward_mean, iteration)
            self.writer.add_scalar("Training/Reward_Std", reward_std, iteration)
            self.writer.add_scalar("Training/LearningRate", current_lr, iteration)
            self.writer.add_scalar("Training/Done_Count", done_count, iteration)
            self.writer.add_scalar(
                "Training/Terminated_Count", terminated_count, iteration
            )  # Number of failures (cartpole falls)
            self.writer.add_scalar(
                "Training/Truncated_Count", truncated_count, iteration
            )  # Number of successes
            self.writer.add_scalar(
                "Training/Ally_Reached_Target_Count", ally_reached_target_count, iteration
            )  # Number of times ally reached target
            self.writer.add_scalar(
                "Training/Explained_Variance", explained_var, iteration
            )  # Explained variance of value function

            # Save MODEL
            if iteration % 100 == 0 and iteration > 0:
                checkpoint = {
                    "iteration": iteration,
                    "policy_state_dict": self.policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_module_state_dict": self.loss_module.state_dict(),
                    "reward_mean": reward_mean,
                    "logs": self.logs,
                    "hyperparameters": {
                        "clip_epsilon": self.clip_epsilon,
                        "entropy_coef": self.entropy_coef,
                        "critic_coef": self.critic_coef,
                        "gamma": self.gamma,
                        "lmbda": self.lmbda,
                        "learning_rate": self.learning_rate,
                        "num_epochs": self.num_epochs,
                        "frames_per_batch": self.frames_per_batch,
                        "minibatch_size": self.minibatch_size,
                    },
                }
                save_dir = "checkpoints"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, f"checkpoint_iter_{iteration}.pt")
                torch.save(checkpoint, save_path)
                print(f"Model saved: {save_path}")


if __name__ == "__main__":

    finetuner = FineTuner()
    finetuner.train()