import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import PPO, SAC
import argparse


def evaluate_saved_model(model_name, model_path, env_name, num_episodes=10, render=True):
    """
    Load a saved model and evaluate it on the environment.
    
    Args:
        model_path: Path to the saved model file (.zip)
        env_name: Name of the environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        Average reward and success rate
    """
    env = gym.make(env_name, render_mode="human" if render else None)
    env.unwrapped.target_in_the_air = True
    if model_name == "ppo":
        model = PPO.load(model_path)
    elif model_name == "sac":
        model = SAC.load(model_path)
    
    rewards = []
    successes = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done and 'is_success' in info:
                successes.append(info['is_success'])
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
    
    mean_reward = np.mean(rewards)
    success_rate = np.mean(successes) if successes else 0
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {mean_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    env.close()
    return mean_reward, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved model")
    parser.add_argument("--env", type=str, default="FetchPickAndPlaceDense-v4", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    model_path = f"./weights/{args.model_name}_{args.env}_seed{args.seed}/{args.model_name}_{args.env}_best_model_seed{args.seed}.zip"
    print(f"Loading model from: {model_path}")
    
    evaluate_saved_model(
        model_name=args.model_name,
        model_path=model_path,
        env_name=args.env,
        num_episodes=args.episodes,
        render=not args.no_render
    )
