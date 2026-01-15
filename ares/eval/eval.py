import torch
import os
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from ares.environment.base_env import BaseEnv
from config.config import config
import time
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.hyperparameters_and_setup import device


def create_simple_env():
    """Create a simple environment without torchRL wrapper."""
    return BaseEnv(config=config)


def load_checkpoint(checkpoint_path, policy):
    """
    Load a saved checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        policy (torch.nn.Module): The policy model to load the state into.

    Returns:
        dict: Loaded checkpoint data.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint non trouvé: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    print(f"Modèle chargé depuis l'itération {checkpoint['iteration']}")
    print(f"Récompense moyenne: {checkpoint['reward_mean']:.4f}")

    return checkpoint


def evaluate_agent(policy, num_episodes=5, max_steps=500, render=True):
    """Evaluate the trained agent."""

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        # Create a simple environment
        env = create_simple_env()

        # Reset environment with classic gym API
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            # Convert observation to tensor
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                obs_dict = TensorDict({"observation": obs_tensor}, batch_size=[1])
                with set_exploration_type(ExplorationType.DETERMINISTIC):
                    action_dict = policy(obs_dict)
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
            f"Épisode terminé: {step_count} steps, Récompense totale: {total_reward:.2f}"
        )
        env.close()


def evaluate_agent_and_record(
    policy, num_episodes=5, max_steps=500, render=True, save_video=False
):
    """Evaluate the trained agent and optionally record video."""

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        # Create a simple environment
        env = create_simple_env()

        # Reset environment with classic gym API
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        frames = []

        while not done and step_count < max_steps:
            # Convert observation to tensor
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                obs_dict = TensorDict({"observation": obs_tensor}, batch_size=[1])
                with set_exploration_type(ExplorationType.DETERMINISTIC):
                    action_dict = policy(obs_dict)
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
            f"Épisode terminé: {step_count} steps, Récompense totale: {total_reward:.2f}"
        )

        # Save video if requested
        if save_video and len(frames) > 0:
            try:
                import imageio
                
                video_path = f"videos/episode_{episode + 1}.mp4"
                os.makedirs("videos", exist_ok=True)
                
                # Use imageio with ffmpeg for better MP4 compatibility
                imageio.mimsave(video_path, frames, fps=30, codec='libx264', quality=8)
                print(f"Video saved to {video_path}")
            except ImportError:
                # Fallback to OpenCV with H.264 codec
                import cv2
                
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


if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_iter_8900.pt"
    # checkpoint_path = "ares/models/1_ally_go_to_target.pt"
    # checkpoint_path = "ares/models/3.1_ally_tries_to_dodge_enemies.pt"
    checkpoint = load_checkpoint(checkpoint_path, policy)

    if checkpoint is not None:
        evaluate_agent(policy, num_episodes=5, max_steps=400, render=True)
        # evaluate_agent_and_record(
        #     policy, num_episodes=5, max_steps=400, render=False, save_video=True
        # )
