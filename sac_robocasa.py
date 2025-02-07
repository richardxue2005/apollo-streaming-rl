import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from streaming_drl.optim import ObGD as Optimizer
from streaming_drl.sparse_init import sparse_init
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from streaming_drl.time_wrapper import AddTimeInfo

from gym_envs import make_lift_env

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
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


def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, render=False):
    torch.manual_seed(seed); np.random.seed(seed)
    
    # Add log file setup
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{env_name}-training_log_seed_{seed}.txt")
    open(log_file, 'w').close()

    env = gym.make(env_name, render=render)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff, overshooting_info)
        s = s_prime
        if terminated or truncated:
            if debug:
                with open(log_file, 'a') as f:
                    f.write(f"Episodic Return: {info['episode']['r'][0]}, Time Step {t}\n")
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()

    # Save model and env data
    save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    os.makedirs(save_dir, exist_ok=True)  
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

    # Save model weights
    save_dir = "stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    os.makedirs(save_dir, exist_ok=True)  
    torch.save(agent.state_dict(), os.path.join(save_dir, "seed_{}.pth".format(seed)))
        
    reward_stats = env.reward_stats
    obs_stats = env.obs_stats
    os.makedirs(save_dir, exist_ok=True)  
    with open(os.path.join(save_dir, "stats_data_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((reward_stats, obs_stats), f)


def test(env_name, seed, lr, gamma, lamda, entropy_coeff, kappa_policy, kappa_value):
    torch.manual_seed(seed)
    env = gym.make(env_name, render=True)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)


    # Load model weights
    agent.load_state_dict(torch.load("stream_ac_Lift-Panda-v0_lr1.0_gamma0.99_lamda0.8_entropy_coeff0.01/seed_{}.pth".format(seed), weights_only=True))
    reward_stats, obs_stats = pickle.load(open("stream_ac_Lift-Panda-v0_lr1.0_gamma0.99_lamda0.8_entropy_coeff0.01/stats_data_{}.pkl".format(seed), "rb"))
    env.obs_stats.mean = obs_stats.mean
    env.obs_stats.var = obs_stats.var
    env.obs_stats.count = obs_stats.count
    env.obs_stats.p = obs_stats.p
    env.reward_stats.mean = reward_stats.mean
    env.reward_stats.var = reward_stats.var
    env.reward_stats.count = reward_stats.count
    env.reward_stats.p = reward_stats.p

    s, _ = env.reset()

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, 2000+1):
        a = np.zeros(7)
        # a = agent.sample_action(s)
        print(f"State: {s}")
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff)
        s = s_prime
        if terminated or truncated:
            cur_return = info['episode']['r'][0]
            print(f"Episode finished with reward: {cur_return}")
            returns.append(cur_return)
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='Lift-Panda-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.render)
    # test(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.entropy_coeff, args.kappa_policy, args.kappa_value)