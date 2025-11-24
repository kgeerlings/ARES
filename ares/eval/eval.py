import torch
import os
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from ares.environment.base_env import BaseEnv
from config.config import config
import time
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.hyperparameters_and_setup import device
import cv2
import numpy as np
from datetime import datetime


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


def setup_video_writer(output_path, fps=30, frame_size=None):
    """
    Setup video writer for recording evaluation.
    
    Args:
        output_path (str): Path to save the video
        fps (int): Frames per second
        frame_size (tuple): Frame size (width, height) - will be determined from first frame if None
    
    Returns:
        cv2.VideoWriter: Video writer object
    """
    # Use H264 codec which is more compatible
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # If no frame size specified, use default
    if frame_size is None:
        frame_size = (640, 480)
    
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def capture_frame(env):
    """
    Capture a frame from the environment for video recording.
    
    Args:
        env: The environment object
        
    Returns:
        np.ndarray: BGR frame as numpy array with correct dimensions
    """
    try:
        # Get the frame from environment rendering
        frame = env.render(mode='rgb_array')
        if frame is not None:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Ensure frame has correct dimensions (height, width, 3)
            if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                return frame_bgr
            else:
                print(f"Frame dimensions incorrectes: {frame_bgr.shape}")
                
    except Exception as e:
        print(f"Erreur lors de la capture de frame: {e}")
    return None


def evaluate_agent(policy, num_episodes=5, max_steps=500, render=True, record_video=False, video_dir="ares/videos"):
    """
    Evaluate the trained agent.
    
    Args:
        policy: The trained policy
        num_episodes (int): Number of episodes to evaluate
        max_steps (int): Maximum steps per episode
        render (bool): Whether to render the environment
        record_video (bool): Whether to record video
        video_dir (str): Directory to save videos
    """
    
    # Create video directory if recording
    if record_video:
        os.makedirs(video_dir, exist_ok=True)

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        # Create a simple environment
        env = create_simple_env()

        # Setup video recording for this episode
        video_writer = None
        video_path = None
        frame_size = None
        
        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(video_dir, f"episode_{episode+1}_{timestamp}.mp4")

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

            # Render and record if requested
            if render or record_video:
                try:
                    if render:
                        env.render()
                    
                    # Capture frame for video recording
                    if record_video:
                        frame = capture_frame(env)
                        if frame is not None:
                            # Initialize video writer with actual frame size on first frame
                            if video_writer is None:
                                frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
                                video_writer = setup_video_writer(video_path, fps=30, frame_size=frame_size)
                                print(f"Enregistrement vidéo: {video_path} - Taille: {frame_size}")
                            
                            # Write frame to video
                            if video_writer is not None:
                                video_writer.write(frame)
                    
                    if render:
                        time.sleep(0.01)  # Pause for rendering
                except Exception as e:
                    print(f"Erreur lors du rendu/enregistrement: {e}")

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
        
        # Clean up video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Vidéo sauvegardée: {video_path}")
        
        env.close()


if __name__ == "__main__":
    # checkpoint_path = "checkpoints/checkpoint_iter_5900.pt"
    checkpoint_path = "ares/models/1_ally_go_to_target.pt"
    checkpoint = load_checkpoint(checkpoint_path, policy)

    if checkpoint is not None:
        print("Début de l'évaluation...")
        evaluate_agent(
            policy=policy, 
            num_episodes=5, 
            max_steps=1000, 
            render=True,
            record_video=True,  
            video_dir="ares/videos/1" 
        )
        print("Évaluation terminée!")
    else:
        print("Impossible de charger le modèle.")
