import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import wandb
import argparse
import time
from wandb.integration.sb3 import WandbCallback


# class CustomEvalCallback(BaseCallback):
#     def __init__(self, model, model_name, eval_env, eval_freq, eval_episodes=50, wandb_log=False, env_name="FetchPickAndPlace-v4", seed=0):
#         super(CustomEvalCallback, self).__init__()
#         self.model = model
#         self.model_name = model_name
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.eval_episodes = eval_episodes
#         self.wandb_log = wandb_log
#         self.best_mean_reward = -float('inf')
#         self.start_time = None
#         self.env_name = env_name
#         self.seed = seed
#         self.converged = False
        
#         log_dir = "logs"
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         self.eval_log_file = os.path.join(log_dir, f"{self.model_name}-{self.env_name}-eval_log_seed_{self.seed}.txt")
#         open(self.eval_log_file, 'w').close()
        
#         self.save_dir = f"./weights/{self.model_name}_{self.env_name}_seed{self.seed}/"
#         os.makedirs(self.save_dir, exist_ok=True)
    
#     def _init_callback(self):
#         self.start_time = time.time()
    
#     def _on_step(self):
#         if self.num_timesteps % self.eval_freq == 0:
#             print(f"Running evaluation at timestep {self.num_timesteps}")
#             returns = []
#             successes = []
            
#             for _ in range(self.eval_episodes):
#                 obs, _ = self.eval_env.reset(seed=self.seed)
#                 done = False
#                 episode_return = 0
#                 episode_success = 0 

#                 while not done:
#                     action, _ = self.model.predict(obs, deterministic=True)
#                     obs, reward, terminated, truncated, info = self.eval_env.step(action)
#                     episode_return += reward
#                     done = terminated or truncated
                    
#                     # Only check success at the final step of the episode
#                     if done and 'is_success' in info:
#                         episode_success = info['is_success']
                
#                 # After episode is done, record results
#                 returns.append(episode_return)
#                 successes.append(episode_success)
            
#             mean_return = sum(returns) / len(returns)
#             success_rate = sum(successes) / len(successes) if successes else 0
            
#             if self.wandb_log:
#                 wandb.log({
#                     "eval/mean_return": mean_return,
#                     "eval/success_rate": success_rate,
#                     "eval/episode": self.num_timesteps // self.eval_freq,
#                 })
            
#             # Check for convergence
#             if success_rate > 0.90 and mean_return > self.best_mean_reward:
#                 elapsed_time = time.time() - self.start_time
#                 if not self.converged:
#                     with open(self.eval_log_file, 'a') as f:
#                         f.write(f"Model converged after {elapsed_time:.2f} seconds\n")
#                     self.converged = True
                
#             if mean_return > self.best_mean_reward:
#                 self.best_mean_reward = mean_return
#                 model_path = os.path.join(self.save_dir, f"{self.model_name}_{self.env_name}_best_model_seed{self.seed}")
#                 self.model.save(model_path)
            
#             with open(self.eval_log_file, 'a') as f:
#                 f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Eval Number: {self.num_timesteps // self.eval_freq}\n")
        
#         return True
    

# class CustomTrainCallback(BaseCallback):
#     def __init__(self, model, env_name, seed, wandb_log=False):
#         super(CustomTrainCallback, self).__init__()
#         self.model_name = model
#         self.wandb_log = wandb_log
#         self.env_name = env_name
#         self.seed = seed
        
#         # Track episodes we've already seen
#         self.logged_episodes = set()
#         self.total_episodes_logged = 0
        
#         log_dir = "logs"
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         self.log_file = os.path.join(log_dir, f"{self.model_name}-{env_name}-training_log_seed_{seed}.txt")
#         open(self.log_file, 'w').close()
    
#     def _init_callback(self):
#         self.logged_episodes = set()
#         self.total_episodes_logged = 0

#     def _on_step(self):
#         if self.num_timesteps % 1000 == 0:
#             print(f"CustomTrainCallback at timestep {self.num_timesteps}, buffer size: {len(self.model.ep_info_buffer) if hasattr(self.model, 'ep_info_buffer') else 'N/A'}")
            
#             with open(self.log_file, 'a') as f:
#                 f.write(f"DEBUG: Timestep {self.num_timesteps}, total_episodes_logged: {self.total_episodes_logged}\n")
        
#         if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
#             # Check for any new episodes in the buffer
#             current_buffer_size = len(self.model.ep_info_buffer)
            
#             current_episode_ids = set()
            
#             for i in range(current_buffer_size):
#                 try:
#                     ep_info = self.model.ep_info_buffer[i]

#                     # Create a unique identifier for this episode (combination of return and length)
#                     episode_return = ep_info.get('r', 0)
#                     episode_length = ep_info.get('l', 0)
#                     episode_id = f"{episode_return:.4f}_{episode_length}"
                    
#                     current_episode_ids.add(episode_id)
#                     if episode_id in self.logged_episodes:
#                         continue
                    
#                     with open(self.log_file, 'a') as f:
#                         f.write(f"Episodic Return: {episode_return:.2f}, Length: {episode_length}, Time Step {self.num_timesteps}\n")
                    
#                     if self.wandb_log:
#                         wandb.log({
#                             "train/episodic_return": episode_return,
#                             "train/episode_length": episode_length,
#                             "train/timesteps": self.num_timesteps,
#                             "train/episodes": self.total_episodes_logged
#                         })
                    
#                     self.logged_episodes.add(episode_id)
#                     self.total_episodes_logged += 1
                    
#                 except Exception as e:
#                     # Catch and log any errors during the logging process
#                     print(f"Error in CustomTrainCallback: {e}")
#                     with open(self.log_file, 'a') as f:
#                         f.write(f"ERROR: {e} at timestep {self.num_timesteps}\n")
            
#             # Clean out episodes that are no longer in the buffer every time
#             # This prevents the logged_episodes set from growing indefinitely
#             self.logged_episodes = self.logged_episodes.intersection(current_episode_ids)
        
#         return True

def train_model(model_type="ppo", env_name="FetchPickAndPlace-v4", total_timesteps=1_000_000, seed=0, n_envs=4, eval_freq=10000, eval_episodes=50, wandb_log=False, table_only=False):
    if wandb_log:
        wandb.init(
            entity="apollo-lab",
            project=f"{model_type.lower()}-test",
            config={
                "env_name": env_name,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
            },
            sync_tensorboard=True,
            save_code=True,
            name=f"{env_name}_seed{seed}_total_timesteps{total_timesteps}_n_envs{n_envs}"
        )
    
    env = make_vec_env(env_name, n_envs=n_envs, seed=seed)
    eval_env = gym.make(env_name)

    if env_name == "FetchPickAndPlace-v4" and table_only:
        # Set target_in_the_air to False for all environments
        for i in range(n_envs):
            if hasattr(env.envs[i].unwrapped, 'target_in_the_air'):
                env.envs[i].unwrapped.target_in_the_air = False
            print(f"Set target_in_the_air=False for env {i}")
    
        if hasattr(eval_env.unwrapped, 'target_in_the_air'):
            eval_env.unwrapped.target_in_the_air = False
            print(f"Set target_in_the_air=False for eval_env")

    if model_type.lower() == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1,
                    learning_rate=5e-4,
                    n_steps=2048 // n_envs,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    policy_kwargs=dict(net_arch=[512, 512]),
                    seed=seed)
    elif model_type.lower() == "sac":
        goal_selection_strategy = GoalSelectionStrategy.FUTURE
        max_steps = env.envs[0].unwrapped.spec.max_episode_steps

        # Create the model with Hindsight Experience Replay
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                env=env,
                goal_selection_strategy=goal_selection_strategy,
                n_sampled_goal=4,
            ),
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=100 * n_envs,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=n_envs,
            ent_coef='auto',
            target_entropy='auto',
            policy_kwargs=dict(net_arch=[512, 512]),
            tensorboard_log=f"logs/{model_type.lower()}_{env_name}_seed{seed}/tensorboard",
            seed=seed
        )
    else:
        # implement TQC
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'ppo' or 'sac'.")

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"weights/{model_type.lower()}_{env_name}_seed{seed}/best",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    all_callbacks = [eval_callback]

    if wandb_log:
        train_callback = WandbCallback(
            model_save_path=f"weights/{model_type.lower()}_{env_name}_seed{seed}/checkpoints",
            model_save_freq=100,
            verbose=2,
        )
        all_callbacks.append(train_callback)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=all_callbacks)
    model.save(f"weights/final/{model_type.lower()}_{env_name}_seed{seed}")
    
    if wandb_log:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ppo")
    parser.add_argument("--env", type=str, default="FetchPushDense-v4")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=2_500_000)
    parser.add_argument("--eval_freq", type=int, default=1_000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    
    args = parser.parse_args()
    
    print(f"Training {args.model_name}...")
    train_model(model_type=args.model_name, env_name=args.env, n_envs=args.n_envs, 
               seed=args.seed, total_timesteps=args.total_timesteps, 
               eval_freq=args.eval_freq, eval_episodes=args.eval_episodes, 
               wandb_log=args.wandb_log, table_only=args.table_only)
    
