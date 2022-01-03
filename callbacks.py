from __future__ import print_function

from typing import List, Dict, Any

import gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class RecorderCallback(BaseCallback):
    def __init__(self, env: gym.Env, eval_env: gym.Env, extra_stats: List, render_freq=1000, n_eval_episodes=3, verbose=1, deterministic=True):
        super(RecorderCallback, self).__init__(verbose)
        self.env = env
        self.extra_stats = extra_stats
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self):
        # Add additional statistics
        env = self.env
        for i, done in enumerate(env.buf_dones):
            if done:
                statistics = env.buf_infos[i]
                for stats in self.extra_stats:
                    if stats not in statistics:
                        self.logger.warn(f'{stats} is not an available statistic')
                        continue
                    self.logger.record_mean(f'rollout/ep_{stats}_mean', statistics[stats])

        # Render video
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            reward, ep_len = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                return_episode_rewards=True
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(np.array([screens])), fps=20),
                exclude=("stdout", "log", "json", "csv"),
            )
        # Plot values (here a random variable)
        # figure = plt.figure()
        # figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        # self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        # plt.close()
        return True
