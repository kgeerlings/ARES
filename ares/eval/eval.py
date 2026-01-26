import os
import time
import imageio
import cv2
import torch
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from ares.environment.base_env import BaseEnv
from ares.environment.dodging_enemies_env_wrapper import DodgingEnemiesEnvWrapper
from ares.environment.go_and_return_env_wrapper import GoAndReturnEnvWrapper
from config.config import config
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.hyperparameters_and_setup import device

class Evaluator:
    """Class to evaluate a trained RL agent."""

    def __init__(self, env_choice, configuration, policy=policy, device=device) -> None:
        """
        Args:
            env_choice (str): Choice of the environment to evaluate (1, 2 or 3).
            config (dict): Configuration for the environment.
            policy (torch.nn.Module): The policy model to evaluate.
            device (torch.device): Device to run the evaluation on.

            Returns:
                None
        """

        self.env_choice = env_choice
        self.config = configuration
        self.policy = policy
        self.device = device


    def _create_env(self):
        """Creates an environment for evaluation."""
        if self.env_choice==1:
            return BaseEnv(config=self.config)
        elif self.env_choice==2:
            return GoAndReturnEnvWrapper(config=self.config)
        elif self.env_choice==3:
            return DodgingEnemiesEnvWrapper(config=self.config)
        else:
            raise ValueError(f"Unknown environment choice: {self.env_choice}")


    @staticmethod
    def _load_checkpoint(checkpoint_path: str, policy: torch.nn.Module) -> dict:
        """
        Load a saved checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            policy (torch.nn.Module): The policy model to load the state into.

        Returns:
            dict: Loaded checkpoint data.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint npt found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        print(f"Model loaded from iteration {checkpoint['iteration']}")
        print(f"Average reward: {checkpoint['reward_mean']:.4f}")

        return checkpoint


    def _evaluate_agent(self, policy, num_episodes=5, max_steps=500, render=True):
        """Evaluate the trained agent."""

        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

            # Create a simple environment
            env = self._create_env()

            # Reset environment with classic gym API
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            done = False

            while not done and step_count < max_steps:
                # Convert observation to tensor
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                # Get action from policy
                with torch.no_grad():
                    obs_dict = TensorDict({"observation": obs_tensor}, batch_size=[1])
                    with set_exploration_type(ExplorationType.DETERMINISTIC):
                        action_dict = self.policy(obs_dict)
                        action = action_dict["action"].squeeze(0).cpu().numpy()

                # Render if requested
                if render:
                    try:
                        env.render()
                        time.sleep(0.01)  # Pause for rendering
                    except:  # noqa: E722
                        pass

                # Make a step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                # Afficher les infos périodiquement
                if step_count == 1 or step_count % 10 == 0:
                    print(
                        f"Step {step_count}: Reward = {reward:.2f}, Total Reward = {total_reward:.2f}"
                    )
                    print(f"Observation: {obs}")

            print(
                f"Episode done in {step_count} steps, Total reward: {total_reward:.2f}"
            )
            env.close()


    def _evaluate_agent_and_record(
        self, policy, num_episodes=5, max_steps=500, render=True, save_video=False
    ):
        """Evaluate the trained agent and optionally record video."""

        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

            # Create a simple environment
            env = self._create_env()

            # Reset environment with classic gym API
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            done = False

            frames = []

            while not done and step_count < max_steps:
                # Convert observation to tensor
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                # Get action from policy
                with torch.no_grad():
                    obs_dict = TensorDict({"observation": obs_tensor}, batch_size=[1])
                    with set_exploration_type(ExplorationType.DETERMINISTIC):
                        action_dict = self.policy(obs_dict)
                        action = action_dict["action"].squeeze(0).cpu().numpy()

                # Make a step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                # Render if requested
                if render or save_video:
                    try:
                        frame = env.render(mode="rgb_array")
                        if save_video and frame is not None:
                            frames.append(frame)
                        if render:
                            env.render(mode="human")
                            time.sleep(0.01)  # Pause for rendering
                    except:  # noqa: E722
                        pass

                # Afficher les infos périodiquement
                if step_count == 1 or step_count % 10 == 0:
                    print(
                        f"Step {step_count}: Reward = {reward:.2f}, Total Reward = {total_reward:.2f}"
                    )
                    print(f"Observation: {obs}")

            print(
                f"Episode done in {step_count} steps, Total reward: {total_reward:.2f}"
            )

            # Save video if requested
            if save_video and len(frames) > 0:
                try:                    
                    video_path = f"videos/episode_{episode + 1}.mp4"
                    os.makedirs("videos", exist_ok=True)
                    
                    # Use imageio with ffmpeg for better MP4 compatibility
                    imageio.mimsave(video_path, frames, fps=30, codec='libx264', quality=8)
                    print(f"Video saved to {video_path}")
                except ImportError:
                    # Fallback to OpenCV with H.264 codec
                    
                    video_path = f"videos/episode_{episode + 1}.mp4"
                    os.makedirs("videos", exist_ok=True)

                    height, width, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

                    for frame in frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    out.release()
                    print(f"Video saved to {video_path} (using OpenCV)")

            env.close()

    def run(self, checkpoint_path: str, num_episodes=5, max_steps=400, record: bool = False, save_video=True) -> None:
        """
        Evaluate the trained agent from a checkpoint and record if it is asked.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            num_episodes (int): Number of episodes to evaluate.
            max_steps (int): Maximum steps per episode.
            record (bool): Whether to record the evaluation.
            save_video (bool): Whether to save video of the evaluation.

        Returns:
            None
        """
        checkpoint = self._load_checkpoint(checkpoint_path, self.policy)

        if checkpoint:
            if record:
                self._evaluate_agent_and_record(
                    self.policy, num_episodes=num_episodes, max_steps=max_steps, render=False, save_video=save_video
                )
            else:
                self._evaluate_agent(self.policy, num_episodes=num_episodes, max_steps=max_steps, render=True)

if __name__ == "__main__":

    eval = Evaluator(3, configuration=config)

    checkpoints = [
        "checkpoints/checkpoint_iter_9900.pt",
        "ares/models/1_ally_go_to_target.pt",
        "ares/models/2_ally_go_to_target_and_return_to_base.pt",
        "ares/models/3_ally_semi_dodges_enemies.pt",
        "ares/models/3.1_ally_tries_to_dodge_enemies.pt",
        "ares/models/3.2_ally_dodges_enemies.pt",]

    checkpoint_path = checkpoints[5]    
    
    eval.run(checkpoint_path, record=False)