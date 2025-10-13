import gym
import numpy as np
from tqdm import tqdm
from ares.entities import Ally

def train_agent_n_episodes(env : gym.Env, agent : Ally, n : int = 20, verbose_freq : int = None) -> dict:
    """Train the agent for a number of episodes.

    Args:
        env (gym.Env): The environment to train the agent in.
        agent (AgentRL): The agent to train.
        n (int, optional): The number of episodes to train the agent for. Defaults to 20.
        verbose_freq (int, optional): Episode frequency to print progress. If None, no progress is printed. Defaults to None.

    Returns:
        dict: A dictionary containing the total rewards and steps taken in each episode.
    """
    results = {
        "total_rewards": [],
        "steps": [],
    }
    
    # Training loop
    for episode in tqdm(range(n)):
        observation, info = env.reset()
        total_reward = 0
        step = 0
        done = False
        while not done:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            agent.learn(observation, action, reward, next_observation, done)
            step += 1
            observation = next_observation
        
        # Log results
        results["total_rewards"].append(total_reward)
        results["steps"].append(step)
        if verbose_freq is not None and (episode + 1) % verbose_freq == 0:
            avg_reward = np.mean(results["total_rewards"][-verbose_freq:])
            avg_steps = np.mean(results["steps"][-verbose_freq:])
            print(f"Episode {episode + 1}/{n} - Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")
    
    return results
