import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import torch.nn.functional as F
from torch.distributions import Normal
from streaming_drl.optim import ObGD as Optimizer
from streaming_drl.sparse_init import sparse_init
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from streaming_drl.time_wrapper import AddTimeInfo
import wandb    
import time
# from gym_envs import make_lift_env

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

class StreamACRunner:
    def __init__(
        self,
        env_name,
        seed=0,
        hidden_size=128,
        lr=1.0,
        gamma=0.99,
        lamda=0.8,
        entropy_coeff=0.01,
        kappa_policy=3.0,
        kappa_value=2.0,
        total_steps=100_000,
        eval_frequency=10_000,
        eval_episodes=50,
        debug=False,
        wandb_log=False,
        overshooting_info=False,
        render=False,
        max_episode_steps=100
    ):
        self.env_name = env_name
        self.seed = seed
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coeff = entropy_coeff
        self.kappa_policy = kappa_policy
        self.kappa_value = kappa_value
        self.total_steps = total_steps
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.debug = debug
        self.wandb_log = wandb_log
        self.overshooting_info = overshooting_info
        self.render = render
        self.max_episode_steps = max_episode_steps
        
        self.agent = None
        self.env = None
        self.log_file = None
        self.eval_log_file = None
        self.returns = []
        self.term_time_steps = []
        
    def create_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{self.env_name}-training_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}.txt")
        open(log_file, 'w').close()

        eval_log_file = os.path.join(log_dir, f"{self.env_name}-eval_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}.txt")
        open(eval_log_file, 'w').close()

        if self.wandb_log:
            wandb.init(
                entity="apollo-lab",
                project=f"stream-ac-test",
                config={
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                    "lambda": self.lamda,
                    "total_steps": self.total_steps,
                    "entropy_coeff": self.entropy_coeff,
                    "kappa_policy": self.kappa_policy,
                    "kappa_value": self.kappa_value,
                    "eval_frequency": self.eval_frequency,
                    "eval_episodes": self.eval_episodes,
                },
                name=f"{self.env_name}_seed{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}"
            )

        self.log_file = log_file
        self.eval_log_file = eval_log_file
        
    def setup_environment(self):
        render_mode = "human" if self.render else None
        env = gym.make(self.env_name, render_mode=render_mode, max_episode_steps=self.max_episode_steps)
        env = self.wrap_environment(env)
        return env
    
    def wrap_environment(self, env):
        """Base environment wrappers. Can be overridden by subclasses."""
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = ScaleReward(env, gamma=self.gamma)
        env = NormalizeObservation(env)
        env = AddTimeInfo(env)
        return env
    
    def create_agent(self, env):
        agent = StreamAC(
            n_obs=env.observation_space.shape[0], 
            n_actions=env.action_space.shape[0], 
            hidden_size=self.hidden_size, 
            lr=self.lr, 
            gamma=self.gamma, 
            lamda=self.lamda, 
            kappa_policy=self.kappa_policy, 
            kappa_value=self.kappa_value
        )
        return agent
    
    def save_model_and_stats(self):
        # Save training data
        save_dir = f"results/data_stream_ac_{self.env.spec.id}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        os.makedirs(save_dir, exist_ok=True)  
        with open(os.path.join(save_dir, f"seed_{self.seed}.pkl"), "wb") as f:
            pickle.dump((self.returns, self.term_time_steps, self.env_name), f)

        # Save model weights
        save_dir = f"weights/stream_ac_{self.env.spec.id}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        os.makedirs(save_dir, exist_ok=True)  
        torch.save(self.agent.state_dict(), os.path.join(save_dir, f"seed_{self.seed}.pth"))
        
        # Save env stats
        reward_wrapper = self.env
        while not isinstance(reward_wrapper, ScaleReward) and hasattr(reward_wrapper, 'env'):
            reward_wrapper = reward_wrapper.env
            
        obs_wrapper = self.env
        while not isinstance(obs_wrapper, NormalizeObservation) and hasattr(obs_wrapper, 'env'):
            obs_wrapper = obs_wrapper.env

        reward_stats = reward_wrapper.reward_stats
        obs_stats = obs_wrapper.obs_stats
        with open(os.path.join(save_dir, f"stats_data_{self.seed}.pkl"), "wb") as f:
            pickle.dump((reward_stats, obs_stats), f)

        # Log final model to wandb
        if self.wandb_log:
            wandb.save(os.path.join(save_dir, f"seed_{self.seed}.pth"))
            wandb.finish()
        
        return save_dir
    
    def evaluate(self):
        torch.manual_seed(self.seed)

        returns = []
        successes = []
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        while episode_count < self.eval_episodes:
            a = self.agent.sample_action(s)
            s_prime, r, terminated, truncated, info = self.env.step(a)
            s = s_prime
            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                returns.append(episode_return)

                is_success = info.get('is_success', 0)
                if isinstance(is_success, (list, np.ndarray)):
                    is_success = is_success[0]
                successes.append(is_success)

                episode_count += 1
                s, _ = self.env.reset()

        success_rate = np.mean(successes)
        return returns, success_rate
    
    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.create_logs()
        self.env = self.setup_environment()
        self.agent = self.create_agent(self.env)
        
        if self.debug:
            print(f"seed: {self.seed}", f"env: {self.env.spec.id}")

        self.returns = []
        self.term_time_steps = []
        
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        start_time = time.time()
        converged = False
        
        for t in range(1, self.total_steps + 1):
            # Run evaluation
            if t % self.eval_frequency == 0:
                eval_returns, success_rate = self.evaluate()
                mean_return = np.mean(eval_returns)

                if self.wandb_log:
                    wandb.log({
                        "eval/mean_return": mean_return,
                        "eval/success_rate": success_rate,
                        "eval/episode": t // self.eval_frequency,
                    })
                
                if success_rate > 0.90 and not converged:
                    converged = True
                    elapsed_time = time.time() - start_time
                    with open(self.eval_log_file, 'a') as f:
                        f.write(f"Model converged after {elapsed_time:.2f} seconds\n")
                
                with open(self.eval_log_file, 'a') as f:
                    f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Eval Number: {t // self.eval_frequency}\n")

            a = self.agent.sample_action(s)
            s_prime, r, terminated, truncated, info = self.env.step(a)
            self.agent.update_params(s, a, r, s_prime, terminated or truncated, self.entropy_coeff, self.overshooting_info)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                
                if self.wandb_log:
                    wandb.log({
                        "train/episode_return": episode_return,
                        "train/episode": episode_count,
                        "timestep": t
                    })
                
                if self.debug:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Episodic Return: {episode_return}, Time Step {t}\n")

                self.returns.append(episode_return)
                self.term_time_steps.append(t)
                terminated, truncated = False, False
                s, _ = self.env.reset()
                episode_count += 1
        
        self.env.close()
        
        with open(self.eval_log_file, 'a') as f:
            f.write(f"Total training time: {time.time() - start_time:.2f} seconds\n")

        # Save model, stats and data
        self.save_model_and_stats()
    
    def test(self, num_episodes=10):
        torch.manual_seed(self.seed)
        render_mode = "human" if self.render else None

        env = gym.make(self.env_name, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        # Load model weights
        save_dir = f"weights/stream_ac_{self.env_name}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        reward_stats, obs_stats = pickle.load(open(os.path.join(save_dir, f"stats_data_{self.seed}.pkl"), "rb"))
        
        env = ScaleReward(env, gamma=self.gamma)
        env.reward_stats.mean = reward_stats.mean
        env.reward_stats.var = reward_stats.var
        env.reward_stats.count = reward_stats.count
        env.reward_stats.p = reward_stats.p

        env = NormalizeObservation(env)
        env.obs_stats.mean = obs_stats.mean
        env.obs_stats.var = obs_stats.var
        env.obs_stats.count = obs_stats.count
        env.obs_stats.p = obs_stats.p

        env = AddTimeInfo(env)

        agent = StreamAC(
            n_obs=env.observation_space.shape[0], 
            n_actions=env.action_space.shape[0], 
            hidden_size=self.hidden_size, 
            lr=self.lr, 
            gamma=self.gamma, 
            lamda=self.lamda, 
            kappa_policy=self.kappa_policy, 
            kappa_value=self.kappa_value
        )
        agent.load_state_dict(torch.load(os.path.join(save_dir, f"seed_{self.seed}.pth")))

        returns = []
        s, _ = env.reset(seed=self.seed)
        episode_count = 0
        
        while episode_count < num_episodes:
            a = agent.sample_action(s)
            s_prime, r, terminated, truncated, info = env.step(a)
            agent.update_params(s, a, r, s_prime, terminated or truncated, self.entropy_coeff)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                print(f"Episode {episode_count + 1} finished with reward: {episode_return}")
                returns.append(episode_return)
                episode_count += 1
                s, _ = env.reset()
        
        env.close()
        print(f"\nAverage return over {num_episodes} episodes: {np.mean(returns):.2f}")


class AntStreamACRunner(StreamACRunner):
    def __init__(self, do_damage, damage_start_step, damage_steps, damage_type, **kwargs):
        kwargs.setdefault('env_name', 'Ant-v5')
        kwargs.setdefault('max_episode_steps', 1000)
        super().__init__(**kwargs)
        self.do_damage = do_damage
        self.damage_start_step = damage_start_step
        self.damage_steps = damage_steps
        self.damage_type = damage_type

        self.damage_active = False
        self.damaged_leg = 0
        self.damaged_joints = []
        self.leg_action_map = {
            0: [0, 4],  # Front right leg (hip, ankle)
            1: [1, 5],  # Front left leg
            2: [2, 6],  # Back right leg
            3: [3, 7],  # Back left leg
        }
    
    def create_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        damage_info = f"_damage_{self.damage_type}" if self.do_damage else ""
        
        # Create a run name that will be used for both logs and videos
        self.run_name = f"{self.env_name}_seed{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}{damage_info}"
        
        log_file = os.path.join(log_dir, f"{self.env_name}-training_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}{damage_info}.txt")
        open(log_file, 'w').close()

        eval_log_file = os.path.join(log_dir, f"{self.env_name}-eval_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}{damage_info}.txt")
        open(eval_log_file, 'w').close()

        if self.wandb_log:
            wandb.init(
                entity="apollo-lab",
                project=f"stream-ac-test",
                config={
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                    "lambda": self.lamda,
                    "total_steps": self.total_steps,
                    "entropy_coeff": self.entropy_coeff,
                    "kappa_policy": self.kappa_policy,
                    "kappa_value": self.kappa_value,
                    "eval_frequency": self.eval_frequency,
                    "eval_episodes": self.eval_episodes,
                    "do_damage": self.do_damage,
                    "damage_type": self.damage_type if self.do_damage else "none",
                    "damage_start_step": self.damage_start_step,
                    "damage_steps": self.damage_steps,
                },
                name=self.run_name
            )

        self.log_file = log_file
        self.eval_log_file = eval_log_file
    
    def apply_damage(self, a):
        """Apply damage to the action if damage is active"""
        if not self.damage_active:
            return a
            
        if self.damage_type == 'broken_leg':
            # Broken leg: Set actions for the damaged joints to zero
            for joint_idx in self.damaged_joints:
                a[joint_idx] = 0.0
                
        elif self.damage_type == 'weak_joint':
            # Weak joint: Reduce the strength of the actions for the damaged joints
            for joint_idx in self.damaged_joints:
                a[joint_idx] *= 0.3  # Only 30% of the original strength
                
        elif self.damage_type == 'stuck_joint':
            # Stuck joint: Set actions to a fixed position
            for joint_idx in self.damaged_joints:
                a[joint_idx] = 0.5  # Stuck at middle position
                
        elif self.damage_type == 'noisy_joint':
            # Noisy joint: Add significant noise to the joint actions
            for joint_idx in self.damaged_joints:
                a[joint_idx] += np.random.normal(0, 0.5)
                a[joint_idx] = np.clip(a[joint_idx], self.env.action_space.low[joint_idx], 
                                    self.env.action_space.high[joint_idx])
        return a
    
    def evaluate(self):
        torch.manual_seed(self.seed)

        returns = []
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        while episode_count < self.eval_episodes:
            a = self.agent.sample_action(s)
            
            if self.damage_active:
                a = self.apply_damage(a)
                
            s_prime, r, terminated, truncated, info = self.env.step(a)
            s = s_prime
            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                returns.append(episode_return)

                episode_count += 1
                s, _ = self.env.reset()

        return returns, 0.0
    
    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.create_logs()
        self.env = self.setup_environment()
        self.agent = self.create_agent(self.env)
        
        if self.debug:
            print(f"seed: {self.seed}", f"env: {self.env.spec.id}")

        self.returns = []
        self.term_time_steps = []
        
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        self.damage_active = False
        
        # Create videos directory with run-specific subdirectory
        videos_dir = os.path.join("videos", self.run_name)
        os.makedirs(videos_dir, exist_ok=True)
        
        for t in range(1, self.total_steps + 1):
            # Run evaluation
            if t % self.eval_frequency == 0:
                eval_returns, _ = self.evaluate()
                mean_return = np.mean(eval_returns)

                if self.wandb_log:
                    wandb.log({
                        "eval/mean_return": mean_return,
                        "eval/episode": t // self.eval_frequency,
                    })
                
                with open(self.eval_log_file, 'a') as f:
                    f.write(f"Mean Eval Episodic Return: {mean_return}, Eval Number: {t // self.eval_frequency}\n")

            # Record video before applying damage
            if self.do_damage and t >= self.damage_start_step - 1:
                # Record pre-damage behavior one step before damage
                if (t + 1) % self.damage_steps == 0 and t + 1 >= self.damage_start_step:
                    # Record pre-damage behavior
                    video_path = os.path.join(videos_dir, f"{self.env_name}_step{t}_pre_damage.mp4")
                    self.record_video(video_path)
                    if self.debug:
                        print(f"Recorded pre-damage behavior at step {t} to {video_path}")
                
                # Apply damage at the exact step
                if t % self.damage_steps == 0 and t >= self.damage_start_step:
                    self.damage_active = True
                    self.damaged_leg = np.random.randint(0, 4)
                    self.damaged_joints = self.leg_action_map[self.damaged_leg]
                    if self.wandb_log:
                        wandb.log({"damage_introduced": self.damage_type, "damaged_leg": self.damaged_leg, "timestep": t})
                    if self.debug:
                        print(f"Introducing {self.damage_type} to leg {self.damaged_leg} at step {t}")
                    
                    # Record post-damage behavior
                    video_path = os.path.join(videos_dir, f"{self.env_name}_step{t}_post_damage_{self.damage_type}_leg{self.damaged_leg}.mp4")
                    self.record_video(video_path)
                    if self.debug:
                        print(f"Recorded post-damage behavior at step {t} to {video_path}")
            
            a = self.agent.sample_action(s)
            
            if self.damage_active:
                a = self.apply_damage(a)
                
            s_prime, r, terminated, truncated, info = self.env.step(a)
            self.agent.update_params(s, a, r, s_prime, terminated or truncated, self.entropy_coeff, self.overshooting_info)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                
                if self.wandb_log:
                    wandb.log({
                        "train/episode_return": episode_return,
                        "train/episode": episode_count,
                        "timestep": t
                    })
                
                if self.debug:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Episodic Return: {episode_return}, Time Step {t}\n")

                self.returns.append(episode_return)
                self.term_time_steps.append(t)
                terminated, truncated = False, False
                s, _ = self.env.reset()
                episode_count += 1
        
        self.env.close()

        # Save model, stats and data
        self.save_model_and_stats()

    def record_video(self, video_path, num_steps=500):
        """Record a video of the agent's behavior"""
        # Create a separate environment for recording
        render_env = gym.make(self.env_name, render_mode="rgb_array")
        render_env = self.wrap_environment(render_env)
        
        # Set up video recorder
        video_recorder = gym.wrappers.RecordVideo(
            render_env, 
            video_folder=os.path.dirname(video_path),
            name_prefix=os.path.basename(video_path).split('.')[0],
            episode_trigger=lambda x: True  # Record every episode
        )
        
        # Reset environment
        s, _ = video_recorder.reset(seed=self.seed)
        
        # Run for a fixed number of steps or until episode ends
        for _ in range(num_steps):
            a = self.agent.sample_action(s)
            
            if self.damage_active and video_path.find("post_damage") != -1:
                a = self.apply_damage(a)
                
            s, _, terminated, truncated, _ = video_recorder.step(a)
            
            if terminated or truncated:
                break
        
        video_recorder.close()
        render_env.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='Ant-v5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=600_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--eval_frequency', type=int, default=10_000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb_log', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    # Add Ant-specific arguments
    parser.add_argument('--do_damage', action='store_true')
    parser.add_argument('--damage_start_step', type=int, default=200_000)
    parser.add_argument('--damage_steps', type=int, default=100_000, help='Steps between damage events (Ant-v5 only)')
    parser.add_argument('--damage_type', type=str, default='broken_leg', 
                        choices=['broken_leg', 'weak_joint', 'stuck_joint', 'noisy_joint'],
                        help='Type of damage to apply (Ant-v5 only)')
    args = parser.parse_args()

    # Create runner based on environment
    if args.env_name == 'Ant-v5':
        runner = AntStreamACRunner(
            seed=args.seed,
            hidden_size=args.hidden_size,
            lr=args.lr,
            gamma=args.gamma,
            lamda=args.lamda,
            total_steps=args.total_steps,
            entropy_coeff=args.entropy_coeff,
            kappa_policy=args.kappa_policy,
            kappa_value=args.kappa_value,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            debug=args.debug,
            wandb_log=args.wandb_log,
            overshooting_info=args.overshooting_info,
            render=args.render,
            # Pass Ant-specific arguments
            do_damage=args.do_damage,
            damage_start_step=args.damage_start_step,
            damage_steps=args.damage_steps,
            damage_type=args.damage_type
        )
    else:
        runner = StreamACRunner(
            env_name=args.env_name,
            seed=args.seed,
            hidden_size=args.hidden_size,
            lr=args.lr,
            gamma=args.gamma,
            lamda=args.lamda,
            total_steps=args.total_steps,
            entropy_coeff=args.entropy_coeff,
            kappa_policy=args.kappa_policy,
            kappa_value=args.kappa_value,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            debug=args.debug,
            wandb_log=args.wandb_log,
            overshooting_info=args.overshooting_info,
            render=args.render
        )
    
    if args.mode == 'train':
        runner.train()
    else:
        runner.test()