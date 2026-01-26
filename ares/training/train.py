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

class Trainer:
    """Class to handle the training of the RL agent."""

    def __init__(self, collector=collector, buffer=buffer, loss_module=loss_module, optimizer=optimizer, policy=policy, critic=critic, frames_per_batch=frames_per_batch, minibatch_size=minibatch_size, clip_epsilon=clip_epsilon, entropy_coef=entropy_coef, critic_coef=critic_coef, gamma=gamma, lmbda=lmbda, max_grad_norm=max_grad_norm, learning_rate=learning_rate, num_epochs=num_epochs) -> None:
        """Initialize the Trainer class."""

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

        # Callbacks
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"runs/ares_runs_ally_dodges_enemies/run_{self.current_time}"
        self.writer = SummaryWriter(self.log_dir)
        self.logs = defaultdict(list)


    def train(self):

        tqdm.format_sizeof = lambda x, divisor=None: f"{x:,}" if divisor else f"{x:5.2f}"
        progress_bar = tqdm(total=self.collector.total_frames, desc=None, unit_scale=True)

        for iteration, batch in enumerate(self.collector):

            batch = self.loss_module.value_estimator(
                batch,
            )
            batch = batch.reshape(-1)
            self.buffer.extend(batch)

            epoch_losses = defaultdict(list)

            for _ in range(self.num_epochs):
                for _ in range(self.frames_per_batch // self.minibatch_size):

                    sample = self.buffer.sample()

                    loss_vals = self.loss_module(sample)
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
                    self.optimizer.zero_grad()

                    # Backward propagation
                    total_loss.backward()

                    # Clip gradient
                    params = self.optimizer.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

                    # Push a step to the optimizer
                    self.optimizer.step()

            if hasattr(self.policy, "step"):
                self.policy.step(batch.numel())

            self.collector.update_policy_weights_()
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

            # Count episodes where ally reached target
            ally_reached_target_count = batch["next", "observation"][:, 2].sum().item()

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

            # Save model every 100 iterations
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

    trainer = Trainer()
    trainer.train()