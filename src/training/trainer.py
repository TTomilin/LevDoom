from threading import Lock
from time import sleep

import numpy as np

from agent import Agent


class AsynchronousTrainer:
    def __init__(self,
                 agent: Agent,
                 decay_epsilon = False,
                 initial_epsilon = 1.0,
                 final_epsilon = 0.001,
                 model_save_freq = 5000,
                 memory_update_freq = 5000,
                 train_report_freq = 1000
                 ):

        self.agent = agent

        if not decay_epsilon:
            self.agent.explore = 0

        # Learning rate
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.agent.epsilon = self.initial_epsilon if decay_epsilon else self.final_epsilon

        # Frequencies
        self.model_save_freq = model_save_freq
        self.memory_update_freq = memory_update_freq
        self.train_report_freq = train_report_freq

    def train(self):

        # Wait for sufficient experience to be collected
        while self.agent.memory.buffer_size < self.agent.observe:
            sleep(1)
        print('Training Started')

        Q_values = []
        losses = []
        train_iteration = 0

        while True:
            train_iteration += 1

            # Update epsilon
            if self.agent.epsilon > self.final_epsilon:
                new_epsilon = self.agent.epsilon - (self.initial_epsilon - self.final_epsilon) / self.agent.explore
                self.agent.epsilon = max(self.final_epsilon, new_epsilon)

            # Train the model
            Q_max, loss = self.agent.train()
            Q_values.append(Q_max)
            losses.append(loss)

            # Print mean Q_max & mean loss
            if not train_iteration % self.train_report_freq:
                print(f'Training Report / Iteration {train_iteration} / Mean Q_max: {np.mean(Q_values):.2f} / Mean Loss: {np.mean(losses):.5f}')
                Q_values = []
                losses = []

            # Store the weights of the model after [model_save_freq] iterations
            if not train_iteration % self.model_save_freq:
                self.agent.save_model()

            # Save the experiences from the replay buffer after [update_experience_freq] iterations
            if not train_iteration % self.memory_update_freq:
                self.agent.memory.save()
