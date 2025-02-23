import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import gymnasium_robotics
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

def create_logs(env_name, seed, hidden_size, lr, gamma, lamda, entropy_coeff):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{env_name}-training_log_seed_{seed}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy{entropy_coeff}.txt")
    open(log_file, 'w').close()

    eval_log_file = os.path.join(log_dir, f"{env_name}-eval_log_seed_{seed}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy{entropy_coeff}.txt")
    open(eval_log_file, 'w').close()

    return log_file, eval_log_file

def train(env_name, seed, hidden_size, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, wandb_log, overshooting_info, eval_frequency, eval_episodes, render=False):
    if wandb_log:
        wandb.init(
            entity="apollo-lab",
            project=f"stream-ac-test",
            config={
                "env_name": env_name,
                "seed": seed,
                "hidden_size": hidden_size,
                "learning_rate": lr,
                "gamma": gamma,
                "lambda": lamda,
                "total_steps": total_steps,
                "entropy_coeff": entropy_coeff,
                "kappa_policy": kappa_policy,
                "kappa_value": kappa_value,
                "eval_frequency": eval_frequency,
                "eval_episodes": eval_episodes,
            },
            name=f"{env_name}_seed{seed}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy{entropy_coeff}"
        )

    torch.manual_seed(seed); np.random.seed(seed)
    
    log_file, eval_log_file = create_logs(env_name, seed, hidden_size, lr, gamma, lamda, entropy_coeff)

    # Create environments
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=100)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    episode_count = 0

    start_time = time.time()
    converged = False
    
    for t in range(1, total_steps + 1):
        # Run evaluation
        if t % eval_frequency == 0:
            eval_returns, success_rate = evaluate(agent, env, eval_episodes, seed)
            mean_return = np.mean(eval_returns)

            if wandb_log:
                wandb.log({
                    "eval/mean_return": mean_return,
                    "eval/success_rate": success_rate,
                    "eval/episode": t // eval_frequency,
                })
            
            # Check for convergence - success rate > 0.90 indicates convergence
            if success_rate > 0.90 and not converged:
                converged = True
                elapsed_time = time.time() - start_time
                with open(eval_log_file, 'a') as f:
                    f.write(f"Model converged after {elapsed_time:.2f} seconds\n")
            
            with open(eval_log_file, 'a') as f:
                f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Eval Number: {t // eval_frequency}\n")

        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff, overshooting_info)
        s = s_prime

        if terminated or truncated:
            episode_return = info['episode']['r']
            if isinstance(episode_return, (list, np.ndarray)):
                episode_return = episode_return[0]
            
            if wandb_log:
                wandb.log({
                    "train/episode_return": episode_return,
                    "train/episode": episode_count,
                    "timestep": t
                })
            
            if debug:
                with open(log_file, 'a') as f:
                    f.write(f"Episodic Return: {episode_return}, Time Step {t}\n")

            returns.append(episode_return)
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
            episode_count += 1
    env.close()
    
    with open(eval_log_file, 'a') as f:
        f.write(f"Total training time: {time.time() - start_time:.2f} seconds\n")

    # Save training data
    save_dir = "results/data_stream_ac_{}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy_coeff{entropy_coeff}".format(env.spec.id, hidden_size, lr, gamma, lamda, entropy_coeff)
    os.makedirs(save_dir, exist_ok=True)  
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

    # Save model weights
    save_dir = "weights/stream_ac_{}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy_coeff{entropy_coeff}".format(env.spec.id, hidden_size, lr, gamma, lamda, entropy_coeff)
    os.makedirs(save_dir, exist_ok=True)  
    torch.save(agent.state_dict(), os.path.join(save_dir, "seed_{}.pth".format(seed)))
    
    # Save env stats
    reward_wrapper = env
    while not isinstance(reward_wrapper, ScaleReward) and hasattr(reward_wrapper, 'env'):
        reward_wrapper = reward_wrapper.env
        
    obs_wrapper = env
    while not isinstance(obs_wrapper, NormalizeObservation) and hasattr(obs_wrapper, 'env'):
        obs_wrapper = obs_wrapper.env

    reward_stats = reward_wrapper.reward_stats
    obs_stats = obs_wrapper.obs_stats
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "stats_data_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((reward_stats, obs_stats), f)

    # Log final model to wandb
    if wandb_log:
        wandb.save(os.path.join(save_dir, f"seed_{seed}.pth"))
        wandb.finish()


def evaluate(agent, env, eval_episodes, seed):
    torch.manual_seed(seed)

    returns = []
    successes = []
    s, _ = env.reset(seed=seed)
    episode_count = 0

    while episode_count < eval_episodes:
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
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
            s, _ = env.reset()
    env.close()

    success_rate = np.mean(successes)
    return returns, success_rate


def test(env_name, seed, hidden_size, lr, gamma, lamda, entropy_coeff, kappa_policy, kappa_value, render=True):
    torch.manual_seed(seed)
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    # Load model weights
    save_dir = f"stream_ac_{env_name}_hidden_size{hidden_size}_lr{lr}_gamma{gamma}_lamda{lamda}_entropy_coeff{entropy_coeff}"
    reward_stats, obs_stats = pickle.load(open(f"{save_dir}/stats_data_{seed}.pkl", "rb"))
    
    env = ScaleReward(env, gamma=gamma)
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

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    agent.load_state_dict(torch.load(f"{save_dir}/seed_{seed}.pth", weights_only=True))

    returns = []
    s, _ = env.reset(seed=seed)
    episode_count = 0
    num_episodes = 10
    while episode_count < num_episodes:
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated, entropy_coeff)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='FetchPickAndPlaceDense-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=20_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--eval_frequency', type=int, default=10_000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb_log', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    train(args.env_name, args.seed, args.hidden_size, args.lr, args.gamma, args.lamda, args.total_steps, 
        args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, 
        args.wandb_log, args.overshooting_info, eval_frequency=args.eval_frequency, 
        eval_episodes=args.eval_episodes, render=args.render)
    
    # test(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.render)