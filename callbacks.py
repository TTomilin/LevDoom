from __future__ import print_function

import os
from typing import List, Dict, Any, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class RecorderCallback(BaseCallback):
    def __init__(self,
                 env: gym.Env,
                 eval_env: gym.Env,
                 extra_stats: List[str],
                 model_save_path: str,
                 eval_freq=1000,
                 render_freq=10000,
                 log_path: Optional[str] = None,
                 n_eval_episodes=3,
                 verbose=1,
                 fps=15,
                 deterministic=False):
        super(RecorderCallback, self).__init__(verbose)
        self.env = env
        self.extra_stats = extra_stats
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._fps = fps
        self._deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

        self.model_save_path = model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.evaluations_health = []
        self.evaluations_kill_count = []
        self.evaluations_ammo_used = []
        self.evaluations_movement = []

    def _on_step(self):
        # Add additional statistics
        env = self.env
        for i, done in enumerate(env.buf_dones):
            if done:
                statistics = env.buf_infos[i]
                fields = self.env.envs[i].env.statistics_fields
                for stats in fields:
                    if stats not in statistics:
                        self.logger.warn(f'{stats} is not an available statistic')
                        continue
                    self.logger.record(f'rollout/{stats}', statistics[stats])

        # Evaluate agent
        if not self.n_calls % self._eval_freq:
            screens = []

            def log_values(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                # TODO Implement for all potential environments
                scenario = self._eval_env.envs[0]
                infos = self._eval_env.buf_infos[0]
                # scenario.store_eval_statistics(infos)
                if self._eval_env.buf_dones[0]:
                    scenario.log_evaluation(self.logger, infos)

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                log_values(_locals, _globals)

                screen = self._eval_env.render(mode="rgb_array")

                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            callback = log_values if self.n_calls % self._render_freq else grab_screens  # Callback for rendering
            reward, ep_len = evaluate_policy(
                self.model,
                self._eval_env,
                callback=callback,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                return_episode_rewards=True,
                warn=True
            )

            # Record the episodes
            if not self.n_calls % self._render_freq:
                self.logger.record(
                    "trajectory/video",
                    Video(th.ByteTensor(np.array([screens])), fps=self._fps),
                    exclude=("stdout", "log", "json", "csv"),
                )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(reward)
                self.evaluations_length.append(ep_len)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(reward), np.std(reward)
            mean_ep_length, std_ep_length = np.mean(ep_len), np.std(ep_len)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                self.logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                self.logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/reward", float(mean_reward))
            if self._eval_env.envs[0].env.display_episode_length:
                self.logger.record("eval/episode_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    self.logger.info("New best mean reward!")
                if self.model_save_path is not None:
                    self.model.save(os.path.join(self.model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

        return True
