import imageio
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
import numpy as np
from wandb.integration.sb3 import WandbCallback
import moviepy.editor as mp


class DamageCallback(BaseCallback):
    """Callback for introducing damage to robotic environments during training."""
    
    def __init__(self, do_damage=False, damage_start_step=100_000, damage_steps=100_000, 
                damage_type="broken_leg", leg_action_map=None, wandb_log=False, verbose=0):
        super(DamageCallback, self).__init__(verbose)
        self.do_damage = do_damage
        self.damage_start_step = damage_start_step
        self.damage_steps = damage_steps
        self.damage_type = damage_type
        self.wandb_log = wandb_log
        
        # Damage state variables
        self.damage_active = False
        self.damaged_leg = 0
        self.damaged_joints = []
        
        # Default leg action map for Ant-v5
        self.leg_action_map = leg_action_map or {
            0: [0, 4],  # Front right leg (hip, ankle)
            1: [1, 5],  # Front left leg
            2: [2, 6],  # Back right leg
            3: [3, 7],  # Back left leg
        }
        
    def _on_step(self):
        if self.do_damage and self.num_timesteps >= self.damage_start_step and self.num_timesteps % self.damage_steps == 0:
            self.damage_active = True
            self.damaged_leg = np.random.randint(0, 4)
            self.damaged_joints = self.leg_action_map[self.damaged_leg]
            
            if self.wandb_log:
                wandb.log({
                    "damage_introduced": self.damage_type, 
                    "damaged_leg": self.damaged_leg, 
                    "timestep": self.num_timesteps
                })
                
            if self.verbose > 0:
                print(f"Introducing {self.damage_type} to leg {self.damaged_leg} at step {self.num_timesteps}")
        
        return True
    
    def get_damage_state(self):
        return {
            "active": self.damage_active,
            "leg": self.damaged_leg,
            "joints": self.damaged_joints,
            "type": self.damage_type
        }


class DamageActionWrapper(gym.Wrapper):
    """Wrapper to modify actions based on damage state."""
    
    def __init__(self, env, damage_callback):
        super(DamageActionWrapper, self).__init__(env)
        self.damage_callback = damage_callback
        
    def step(self, action):
        damage_state = self.damage_callback.get_damage_state()
        
        if damage_state["active"]:
            action = self.apply_damage(action, damage_state["type"], damage_state["joints"])
            
        return self.env.step(action)
    
    def apply_damage(self, a, damage_type, damaged_joints):
        """Apply different types of damage to the action."""
        a_modified = a.copy()
        
        if damage_type == 'broken_leg':
            # Broken leg: Set actions for the damaged joints to zero
            for joint_idx in damaged_joints:
                a_modified[joint_idx] = 0.0
                
        elif damage_type == 'weak_joint':
            # Weak joint: Reduce the strength of the actions for the damaged joints
            for joint_idx in damaged_joints:
                a_modified[joint_idx] *= 0.3  # Only 30% of the original strength
                
        elif damage_type == 'stuck_joint':
            # Stuck joint: Set actions to a fixed position
            for joint_idx in damaged_joints:
                a_modified[joint_idx] = 0.5  # Stuck at middle position
                
        elif damage_type == 'noisy_joint':
            # Noisy joint: Add significant noise to the joint actions
            for joint_idx in damaged_joints:
                a_modified[joint_idx] += np.random.normal(0, 0.5)
                a_modified[joint_idx] = np.clip(a_modified[joint_idx], 
                                             self.action_space.low[joint_idx], 
                                             self.action_space.high[joint_idx])
        
        return a_modified


class EvalLogCallback(BaseCallback):
    """Callback for logging evaluation results to file."""
    
    def __init__(self, eval_env, eval_freq, n_eval_episodes, log_file, wandb_log=False, verbose=0):
        super(EvalLogCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_file = log_file
        self.wandb_log = wandb_log
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            episode_rewards = []
            episode_successes = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    
                    if done:
                        episode_rewards.append(episode_reward)
                        if 'is_success' in info:
                            episode_successes.append(info['is_success'])
            
            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes) if episode_successes else 0.0
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"Mean Eval Episodic Return: {mean_reward:.2f}, Success Rate: {success_rate:.2f}, Timestep: {self.num_timesteps}, Eval Number: {self.n_calls // self.eval_freq}\n")
            
            # Log to wandb
            if self.wandb_log:
                wandb.log({
                    "eval/mean_return": mean_reward,
                    "eval/success_rate": success_rate,
                    "eval/episode": self.n_calls // self.eval_freq,
                    "timestep": self.num_timesteps
                })
                
            if self.verbose > 0:
                print(f"Evaluation at timestep {self.num_timesteps}: Mean reward = {mean_reward:.2f}, Success rate = {success_rate:.2f}")
        
        return True


class TrainLogCallback(BaseCallback):
    """Callback for logging training episode results to file."""
    
    def __init__(self, log_file, wandb_log=False, verbose=0):
        super(TrainLogCallback, self).__init__(verbose)
        self.log_file = log_file
        self.wandb_log = wandb_log
        self.episode_count = 0
        
    def _on_step(self):
        # Check if episode is completed in any environment
        for info in self.locals['infos']:
            if 'episode' in info:
                episode_return = info['episode']['r']
                episode_length = info['episode']['l']
                
                # Log to file
                with open(self.log_file, 'a') as f:
                    f.write(f"Episodic Return: {episode_return:.2f}, Length: {episode_length}, Time Step: {self.num_timesteps}\n")
                
                # Log to wandb
                if self.wandb_log:
                    wandb.log({
                        "train/episode_return": episode_return,
                        "train/episode_length": episode_length,
                        "train/episode": self.episode_count,
                        "timestep": self.num_timesteps
                    })
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: return = {episode_return:.2f}, length = {episode_length}")
                
                self.episode_count += 1
        
        return True
    

class DamageVideoCallback(BaseCallback):
    """Callback for recording videos before and after damage is applied."""
    
    def __init__(self, damage_callback, damage_start_step, damage_steps, env_name, model, eval_env, 
                 model_type="sac", seed=0, wandb_log=False, verbose=0):
        super(DamageVideoCallback, self).__init__(verbose)
        self.damage_callback = damage_callback
        self.damage_start_step = damage_start_step
        self.damage_steps = damage_steps
        self.env_name = env_name
        self.model = model
        self.eval_env = eval_env
        self.model_type = model_type
        self.seed = seed
        self.wandb_log = wandb_log
        
        # Create a run name that will be used for videos
        damage_info = f"_damage_{damage_callback.damage_type}" if damage_callback.do_damage else ""
        self.run_name = f"{self.env_name}_{self.model_type}_seed{self.seed}{damage_info}"
        
        # Create videos directory
        self.videos_dir = os.path.join("videos", self.run_name)
        os.makedirs(self.videos_dir, exist_ok=True)
        
    def _on_step(self):
        # Record pre-damage behavior one step before damage
        if (self.damage_callback.do_damage and 
            self.num_timesteps + 1 >= self.damage_start_step and 
            (self.num_timesteps + 1) % self.damage_steps == 0):
            
            # Record video before damage
            video_before_path = os.path.join(self.videos_dir, 
                                           f"{self.env_name}_step{self.num_timesteps+1}_pre_damage.mp4")
            self.record_video(video_before_path, apply_damage=False)
            
            if self.verbose > 0:
                print(f"Recorded pre-damage behavior at step {self.num_timesteps} to {video_before_path}")
        
        # Apply and record damage at the exact step
        if (self.damage_callback.do_damage and 
            self.num_timesteps >= self.damage_start_step and 
            self.num_timesteps % self.damage_steps == 0):
            
            damage_state = self.damage_callback.get_damage_state()
            
            # Record video after damage
            video_after_path = os.path.join(self.videos_dir, 
                                          f"{self.env_name}_step{self.num_timesteps}_post_damage_{damage_state['type']}_leg{damage_state['leg']}.mp4")
            self.record_video(video_after_path, apply_damage=True)
            
            if self.wandb_log:
                wandb.log({
                    "timestep": self.num_timesteps
                })
                
            if self.verbose > 0:
                print(f"Recorded post-damage behavior at step {self.num_timesteps} to {video_after_path}")
        
        return True
    
    def record_video(self, video_path, apply_damage=False, num_steps=500):
        """Record a video of the agent in the environment."""
        # Create a separate environment for recording
        render_env = gym.make(self.env_name, render_mode="rgb_array")
        
        # Apply damage wrapper if needed
        if apply_damage:
            render_env = DamageActionWrapper(render_env, self.damage_callback)
        
        # Set up video recorder
        video_recorder = gym.wrappers.RecordVideo(
            render_env, 
            video_folder=os.path.dirname(video_path),
            name_prefix=os.path.basename(video_path).split('.')[0],
            episode_trigger=lambda x: True  # Record every episode
        )
        
        # Reset environment
        obs, _ = video_recorder.reset(seed=0)
        
        # Run for a fixed number of steps or until episode ends
        for _ in range(num_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = video_recorder.step(action)
            
            if terminated or truncated:
                break
        
        video_recorder.close()
        render_env.close()


class BaseModelTrainer:
    """Base class for training RL models."""
    
    def __init__(
        self,
        model_type="sac",
        env_name="FetchPickAndPlace-v4",
        total_timesteps=1_000_000,
        seed=0,
        n_envs=4,
        eval_freq=10000,
        eval_episodes=50,
        wandb_log=False
    ):
        self.model_type = model_type.lower()
        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.n_envs = n_envs
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.wandb_log = wandb_log
        
        # Will be initialized during setup
        self.env = None
        self.eval_env = None
        self.model = None
        self.callbacks = []
        
        # Log files
        self.log_file = None
        self.eval_log_file = None
        self.returns = []
        self.term_time_steps = []
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.wandb_log:
            config = {
                "env_name": self.env_name,
                "seed": self.seed,
                "total_timesteps": self.total_timesteps,
                "n_envs": self.n_envs,
                "model_type": self.model_type,
            }
            
            # Add any subclass-specific config
            extra_config = self.get_wandb_config()
            if extra_config:
                config.update(extra_config)
            
            wandb.init(
                entity="apollo-lab",
                project=f"{self.model_type}-test",
                config=config,
                sync_tensorboard=True,
                save_code=True,
                name=f"{self.env_name}_seed{self.seed}_total_timesteps{self.total_timesteps}_n_envs{self.n_envs}"
            )
    
    def get_wandb_config(self):
        """Get additional WandB config parameters from subclasses."""
        return {}
    
    def setup_environments(self):
        """Create and configure training and evaluation environments."""
        self.env = make_vec_env(self.env_name, n_envs=self.n_envs, seed=self.seed)
        self.eval_env = gym.make(self.env_name)
        
        # Apply any environment-specific configurations
        self.configure_environments()
    
    def configure_environments(self):
        """Configure environments (to be implemented by subclasses)."""
        pass
    
    def create_model(self):
        """Create the RL model based on specified type."""
        if self.model_type == "ppo":
            self.model = PPO(
                "MultiInputPolicy", 
                self.env, 
                verbose=1,
                learning_rate=5e-4,
                n_steps=2048 // self.n_envs,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[512, 512]),
                seed=self.seed
            )
        elif self.model_type == "sac":
            # Check if the environment is goal-based (like Fetch environments)
            if "Fetch" in self.env_name or hasattr(self.env, "compute_reward"):
                goal_selection_strategy = GoalSelectionStrategy.FUTURE
                
                self.model = SAC(
                    "MultiInputPolicy",
                    self.env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                        env=self.env,
                        goal_selection_strategy=goal_selection_strategy,
                        n_sampled_goal=4,
                    ),
                    verbose=1,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=100 * self.n_envs,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=self.n_envs,
                    ent_coef='auto',
                    target_entropy='auto',
                    policy_kwargs=dict(net_arch=[512, 512]),
                    tensorboard_log=f"logs/{self.model_type}_{self.env_name}_seed{self.seed}/tensorboard",
                    seed=self.seed
                )
            else:
                # For non-goal environments like Ant, use standard replay buffer
                self.model = SAC(
                    "MlpPolicy",  # Use MlpPolicy for non-dict observation spaces
                    self.env,
                    verbose=1,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=100 * self.n_envs,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=self.n_envs,
                    ent_coef='auto',
                    target_entropy='auto',
                    policy_kwargs=dict(net_arch=[512, 512]),
                    tensorboard_log=f"logs/{self.model_type}_{self.env_name}_seed{self.seed}/tensorboard",
                    seed=self.seed
                )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Choose 'ppo' or 'sac'.")
    
    def create_logs(self):
        """Create log files for training and evaluation."""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{self.env_name}-{self.model_type}_training_log_seed_{self.seed}.txt")
        open(log_file, 'w').close()

        eval_log_file = os.path.join(log_dir, f"{self.env_name}-{self.model_type}_eval_log_seed_{self.seed}.txt")
        open(eval_log_file, 'w').close()
        
        self.log_file = log_file
        self.eval_log_file = eval_log_file
    
    def setup_callbacks(self):
        """Create and configure training callbacks."""
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=f"weights/{self.model_type}_{self.env_name}_seed{self.seed}/best",
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
        )
        self.callbacks.append(eval_callback)
        
        # Custom evaluation logging callback
        eval_log_callback = EvalLogCallback(
            eval_env=self.eval_env,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episodes,
            log_file=self.eval_log_file,
            wandb_log=self.wandb_log,
            verbose=1
        )
        self.callbacks.append(eval_log_callback)
        
        # Training logging callback
        train_log_callback = TrainLogCallback(
            log_file=self.log_file,
            wandb_log=self.wandb_log,
            verbose=1
        )
        self.callbacks.append(train_log_callback)
        
        # WandB callback
        if self.wandb_log:
            wandb_callback = WandbCallback(
                model_save_path=f"weights/{self.model_type}_{self.env_name}_seed{self.seed}/checkpoints",
                model_save_freq=100,
                verbose=2,
            )
            self.callbacks.append(wandb_callback)
        
        # Add any environment-specific callbacks
        self.add_custom_callbacks()
    
    def add_custom_callbacks(self):
        """Add environment-specific callbacks (to be implemented by subclasses)."""
        pass
    
    def _configure_env_for_recording(self, env_instance):
        """
        Applies necessary wrappers to an environment instance for recording/testing.
        Subclasses should override if specific wrappers were applied to self.eval_env 
        and are needed for the model to function correctly.
        """
        # Default: no extra wrappers. 
        # Assumes model can handle raw observations from gym.make() or that
        # gym.make() itself provides a sufficiently wrapped environment.
        return env_instance

    def test(self, num_episodes=10, model_load_path=None, record_gif=False):
        """
        Tests a trained model, with an option to record a GIF of the first episode.
        """
        if self.eval_env is None:
            print("Setting up environments for testing...")
            self.setup_environments()

        if model_load_path is None:
            model_load_path = f"weights/final/{self.model_type}_{self.env_name}_seed{self.seed}.zip"
            print(f"No --model_load_path specified, attempting to load from default: {model_load_path}")

        if not os.path.exists(model_load_path):
            print(f"Error: Model file not found at {model_load_path}. Cannot run test.")
            return

        print(f"Loading model from {model_load_path}")
        if self.model_type == "ppo":
            self.model = PPO.load(model_load_path, env=self.eval_env, device="auto")
        elif self.model_type == "sac":
            self.model = SAC.load(model_load_path, env=self.eval_env, device="auto")
        else:
            raise ValueError(f"Unsupported model type for loading: {self.model_type}")
        print("Model loaded successfully.")

        if self.wandb_log and wandb.run is None:
            print("Initializing WandB for test run...")
            test_config = {
                "env_name": self.env_name,
                "seed": self.seed,
                "model_type": self.model_type,
                "model_load_path": model_load_path,
                "num_test_episodes": num_episodes,
                "record_gif": record_gif,
            }
            if hasattr(self, 'get_wandb_config'):
                test_config.update(self.get_wandb_config())

            wandb.init(
                entity="apollo-lab", 
                project=f"{self.model_type}-evaluation",
                config=test_config,
                name=f"eval_{self.env_name}_seed{self.seed}_{os.path.basename(model_load_path)}",
                job_type="evaluation",
                reinit=True 
            )

        episode_rewards = []
        episode_successes = []
        gif_recorded_this_run = False

        video_main_dir = "videos_test_gifs" 
        sanitized_env_name = self.env_name.replace("/", "-").replace("\\", "-")
        video_subdir_name = f"{sanitized_env_name}_{self.model_type}_seed{self.seed}"
        video_subdir = os.path.join(video_main_dir, video_subdir_name)
        os.makedirs(video_subdir, exist_ok=True)

        for i in range(num_episodes):
            obs, info_reset = self.eval_env.reset(seed=self.seed + i) # Use consistent seed for main eval
            done = False
            current_episode_reward = 0
            ep_len = 0
            
            if record_gif and i == 0 and not gif_recorded_this_run:
                print(f"Recording first test episode for GIF in {video_subdir}...")
                
                record_env_instance = gym.make(self.env_name, render_mode="rgb_array")
                record_env_wrapped = self._configure_env_for_recording(record_env_instance)

                temp_video_name_prefix = f"test_episode_s{self.seed}_e{i}_temp_video"
                
                video_recorder_env = gym.wrappers.RecordVideo(
                    record_env_wrapped,
                    video_folder=video_subdir,
                    name_prefix=temp_video_name_prefix,
                    episode_trigger=lambda ep_id: ep_id == 0,
                    disable_logger=True
                )
                
                obs_rec, _ = video_recorder_env.reset(seed=self.seed + i) # Use same seed for this episode
                done_rec = False
                while not done_rec:
                    action_rec, _ = self.model.predict(obs_rec, deterministic=True)
                    obs_rec, _, terminated_rec, truncated_rec, _ = video_recorder_env.step(action_rec)
                    done_rec = terminated_rec or truncated_rec
                
                video_recorder_env.close()
                # record_env_wrapped.close() # RecordVideo closes the wrapped env

                mp4_filename = f"{temp_video_name_prefix}-episode-0.mp4"
                mp4_path = os.path.join(video_subdir, mp4_filename)
                
                gif_filename = f"test_episode_s{self.seed}_{self.env_name}_{self.model_type}.gif"
                gif_path = os.path.join(video_subdir, gif_filename)

                if os.path.exists(mp4_path):
                    try:
                        print(f"Converting {mp4_path} to {gif_path}...")
                        clip = mp.VideoFileClip(mp4_path)
                        clip.write_gif(gif_path, fps=15, logger=None) # Added logger=None to suppress prog_bar
                        clip.close()
                        os.remove(mp4_path) 
                        print(f"Saved test episode GIF to {gif_path}")
                        if self.wandb_log and wandb.run is not None:
                             wandb.log({"test/episode_gif": wandb.Video(gif_path, fps=15, format="gif")})
                    except Exception as e:
                        print(f"Error converting MP4 to GIF: {e}. MP4 kept at {mp4_path}")
                else:
                    print(f"Warning: MP4 video not found at {mp4_path}. Cannot create GIF.")
                gif_recorded_this_run = True

            # Standard evaluation loop using self.eval_env
            info = info_reset
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info_step = self.eval_env.step(action)
                info = info_step # Use the latest info
                done = terminated or truncated
                current_episode_reward += reward
                ep_len += 1
            
            episode_rewards.append(current_episode_reward)
            if 'is_success' in info:
                episode_successes.append(float(info['is_success']))
            
            print(f"Test Episode {i+1}/{num_episodes}: Reward={current_episode_reward:.2f}, Length={ep_len}", end="")
            if 'is_success' in info:
                print(f", Success: {info['is_success']}")
            else:
                print()

        mean_reward = np.mean(episode_rewards)
        print(f"\nMean reward over {num_episodes} test episodes: {mean_reward:.2f}")
        log_data = {"test/mean_reward": mean_reward, "test/num_episodes": num_episodes}

        if episode_successes:
            mean_success_rate = np.mean(episode_successes)
            print(f"Mean success rate: {mean_success_rate:.2f}")
            log_data["test/mean_success_rate"] = mean_success_rate
        
        if self.wandb_log and wandb.run is not None:
            wandb.log(log_data)
    
    def train(self):
        """Run the full training process."""
        # Setup components
        self.setup_wandb()
        self.create_logs()
        self.setup_environments()
        self.create_model()
        self.setup_callbacks()
        
        # Record start time
        start_time = time.time()
        
        # Train the model
        print(f"Training {self.model_type} on {self.env_name}...")
        self.model.learn(
            total_timesteps=self.total_timesteps, 
            progress_bar=True, 
            callback=self.callbacks
        )
        
        # Log total training time
        training_time = time.time() - start_time
        with open(self.eval_log_file, 'a') as f:
            f.write(f"Total training time: {training_time:.2f} seconds\n")
        
        # Save the final model
        save_path = f"weights/final/{self.model_type}_{self.env_name}_seed{self.seed}"
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        # Finish wandb run
        if self.wandb_log:
            wandb.finish()


class FetchPickAndPlaceTrainer(BaseModelTrainer):
    """Trainer for FetchPickAndPlace environment with table-only option."""
    
    def __init__(self, table_only=False, **kwargs):
        # Set default environment name
        kwargs.setdefault('env_name', 'FetchPickAndPlace-v4')
        super().__init__(**kwargs)
        self.table_only = table_only
    
    def get_wandb_config(self):
        return {"table_only": self.table_only}
    
    def configure_environments(self):
        if self.table_only:
            for i in range(self.n_envs):
                if hasattr(self.env.envs[i].unwrapped, 'target_in_the_air'):
                    self.env.envs[i].unwrapped.target_in_the_air = False
                    print(f"Set target_in_the_air=False for env {i}")
            
            if hasattr(self.eval_env.unwrapped, 'target_in_the_air'):
                self.eval_env.unwrapped.target_in_the_air = False
                print(f"Set target_in_the_air=False for eval_env")


class AntTrainer(BaseModelTrainer):
    """Trainer for Ant-v5 environment with damage functionality."""
    
    def __init__(
        self,
        do_damage=False,
        damage_start_step=100_000,
        damage_steps=100_000,
        damage_type="broken_leg",
        **kwargs
    ):
        kwargs.setdefault('env_name', 'Ant-v5')
        super().__init__(**kwargs)
        
        # Damage parameters
        self.do_damage = do_damage
        self.damage_start_step = damage_start_step
        self.damage_steps = damage_steps
        self.damage_type = damage_type
        
        self.damage_callback = None
    
    def get_wandb_config(self):
        return {
            "do_damage": self.do_damage,
            "damage_type": self.damage_type if self.do_damage else None,
            "damage_start_step": self.damage_start_step if self.do_damage else None,
            "damage_steps": self.damage_steps if self.do_damage else None,
        }
    
    def configure_environments(self):
        if self.do_damage:
            # Create damage callback
            self.damage_callback = DamageCallback(
                do_damage=self.do_damage,
                damage_start_step=self.damage_start_step,
                damage_steps=self.damage_steps,
                damage_type=self.damage_type,
                wandb_log=self.wandb_log,
                verbose=1
            )
            
            for i in range(self.n_envs):
                self.env.envs[i] = DamageActionWrapper(self.env.envs[i], self.damage_callback)
            
            self.eval_env = DamageActionWrapper(self.eval_env, self.damage_callback)
    
    def add_custom_callbacks(self):
        if self.damage_callback is not None:
            self.callbacks.append(self.damage_callback)
            
            video_callback = DamageVideoCallback(
                damage_callback=self.damage_callback,
                damage_start_step=self.damage_start_step,
                damage_steps=self.damage_steps,
                env_name=self.env_name,
                model=self.model,
                eval_env=self.eval_env,
                model_type=self.model_type,
                seed=self.seed,
                wandb_log=self.wandb_log,
                verbose=1
            )
            self.callbacks.append(video_callback)

    def _configure_env_for_recording(self, env_instance):
        """Applies DamageActionWrapper if damage is configured for Ant."""
        if self.do_damage and self.damage_callback:
            print("AntTrainer: Wrapping recording/test env with DamageActionWrapper.")
            return DamageActionWrapper(env_instance, self.damage_callback)
        return super()._configure_env_for_recording(env_instance)


def create_trainer(
    model_type="sac", 
    env_name="FetchPickAndPlace-v4", 
    **kwargs
):
    """Factory function to create the appropriate trainer based on environment."""
    if "Ant" in env_name:
        return AntTrainer(model_type=model_type, env_name=env_name, **kwargs)
    elif "Fetch" in env_name:
        return FetchPickAndPlaceTrainer(model_type=model_type, env_name=env_name, **kwargs)
    else:
        return BaseModelTrainer(model_type=model_type, env_name=env_name, **kwargs)


def train_model(
    model_type="sac", 
    env_name="Ant-v5", 
    total_timesteps=1_000_000, 
    seed=0, 
    n_envs=4, 
    eval_freq=10000, 
    eval_episodes=50, 
    wandb_log=False, 
    **env_kwargs
):
    """Function to create and run a model trainer with the specified parameters."""
    trainer = create_trainer(
        model_type=model_type,
        env_name=env_name,
        total_timesteps=total_timesteps,
        seed=seed,
        n_envs=n_envs,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        wandb_log=wandb_log,
        **env_kwargs
    )
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sac")
    parser.add_argument("--env", type=str, default="FetchReachDense-v4")
    parser.add_argument("--n_envs", type=int, default=1) # set to 1 for direct comparison with stream_ac.py
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes for evaluation callback during training")
    parser.add_argument("--wandb_log", action="store_true")
    
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Run mode: train or test")
    parser.add_argument("--model_load_path", type=str, default=None, help="Path to load model for testing (e.g., weights/final/model.zip)")
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of episodes to run during testing")
    parser.add_argument("--record_gif", action="store_true", help="Record a GIF of the first test episode")

    base_args, remaining_args = parser.parse_known_args()
    
    env_specific_parser = argparse.ArgumentParser()
    
    if "Ant" in base_args.env:
        env_specific_parser.add_argument("--do_damage", action="store_true", 
                                        help="Enable damage to the Ant environment")
        env_specific_parser.add_argument("--damage_start_step", type=int, default=200_000, 
                                        help="Step to start introducing damage")
        env_specific_parser.add_argument("--damage_steps", type=int, default=100_000, 
                                        help="Steps between damage events")
        env_specific_parser.add_argument("--damage_type", type=str, default="broken_leg", 
                                        choices=["broken_leg", "weak_joint", "stuck_joint", "noisy_joint"],
                                        help="Type of damage to apply")
    elif "Fetch" in base_args.env:
        env_specific_parser.add_argument("--table_only", action="store_true",
                                        help="Keep target on the table for FetchPickAndPlace")
    
    env_args = env_specific_parser.parse_args(remaining_args)
    
    all_args = vars(base_args)
    all_args.update(vars(env_args))
    
    mode = all_args.pop("mode")
    model_load_path_arg = all_args.pop("model_load_path", None)
    test_episodes_arg = all_args.pop("test_episodes", 10)
    record_gif_arg = all_args.pop("record_gif", False)

    trainer_init_kwargs = {
        "model_type": all_args.pop("model_name"),
        "env_name": all_args.pop("env"),
        "total_timesteps": all_args.pop("total_timesteps"), 
        "seed": all_args.pop("seed"),
        "n_envs": all_args.pop("n_envs"),                  
        "eval_freq": all_args.pop("eval_freq"),            
        "eval_episodes": all_args.pop("eval_episodes"),    
        "wandb_log": all_args.pop("wandb_log"),
    }
    trainer_init_kwargs.update(all_args)

    trainer = create_trainer(**trainer_init_kwargs)
    
    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test(
            num_episodes=test_episodes_arg,
            model_load_path=model_load_path_arg,
            record_gif=record_gif_arg
        )
        if trainer.wandb_log and wandb.run is not None:
            print("Finishing WandB run after testing.")
            wandb.finish()
    
