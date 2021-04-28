from typing import Tuple, List

import os
import pickle
import random  # Handling random number generation
import itertools
import numpy as np
from collections import deque


class ExperienceReplay:
    PER_e = 0.01  # Avoid some experiences to have 0 probability of being taken
    PER_a = 0.6   # Make a trade-off between random sampling and only taking high priority exp
    PER_b = 0.4   # Importance-sampling, from initial value increasing to 1
    PER_b_increment = 0.001  # Importance-sampling increment per sampling

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, experience_path: str, save = False, prioritized = False, capacity = 10000,
                 storage_size = 5000, n_last_samples_included = 3):
        """
        Experience replay buffer based on the collections.deque
        ;param experience_path: Path in which to load the experience from and store it in
        :param save: Flag to control the saving of the replay buffer
        :param prioritized: Flag to select prioritized experience replay instead of epsilon-greedy
        :param capacity: The size of the buffer, i.e. the number of last transitions to save
        :param storage_size: How many transitions are saved to the disk
        :param n_last_samples_included: Number of most recent observations included into the sample batch
        """
        self.capacity = capacity
        self.storage_size = storage_size
        self.include_last = n_last_samples_included
        self.experience_path = experience_path
        self.prioritized = prioritized
        self.save_experience = save
        self.buffer = SumTree(capacity) if prioritized else deque(maxlen = capacity)

    def add(self, experience: Tuple) -> None:
        """
        Store the transition in a replay buffer
        Pop the leftmost transitions in case the experience replay capacity is breached
        :param experience: Transition (<s, a, r, s', t>)
        :return: None
        """
        if not self.prioritized:
            self.buffer.appendleft(experience)
            if self.buffer_size > self.capacity:
                self.buffer.pop()
            return

        # Find the max priority
        max_priority = np.max(self.buffer.tree[-self.buffer.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.buffer.add(max_priority, experience)  # set the max p for new p
        if self.buffer_size > self.capacity:
            self.buffer.popleft()

    def sample(self, batch_size: int, trace_length = 1) -> List:
        """
        Sample a batch of transitions from replay buffer
        :param batch_size: size of the sampled batch
        :param trace_length: length of the experience trace
        :return: tuple of ndarrays with batch_size as first dimension
        """
        if not self.prioritized:
            if trace_length > 1:
                points = random.sample(range(0, self.buffer_size - trace_length), batch_size)
                batch = [list(itertools.islice(self.buffer, point, point + trace_length)) for point in points]
                for trace in batch:
                    trace.reverse()
                batch = np.array(batch)
            else:
                batch = random.sample(self.buffer, batch_size)
                for i in range(self.include_last):
                    batch[i] = self.buffer[i]
            return batch

        n = batch_size  # TODO verify

        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype = np.int32), np.empty((n, 1), dtype = np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.buffer.total_priority / n  # priority segment

        # Increase the PER_b each time a new minibatch is sampled
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.buffer.tree[-self.buffer.capacity:]) / self.buffer.total_priority
        max_weight = 0 if p_min == 0 else (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sampled from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that corresponds to each value is retrieved
            """
            index, priority, data = self.buffer.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.buffer.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            memory_b.append(data)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities in the tree
    """

    def batch_update(self, tree_idx: int, abs_errors: float):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.buffer.update(ti, p)

    def load(self) -> None:
        """
        Load the experiences from external storage into local memory
        :return: None
        """
        print("Loading experiences into replay memory...")
        self.buffer = pickle.load(open(self.experience_path, 'rb'))
        if not self.prioritized:
            self.buffer = deque(self.buffer)

    def save(self) -> None:
        """
        Save the [storage_size] experiences the into external storage
        to be able to continue training when restarting the program
        :return: None
        """
        if not self.save_experience:
            return
        print("Saving experiences from replay memory...")
        if os.path.exists(self.experience_path):
            os.remove(self.experience_path)
        if self.prioritized:
            stored_experience = self.buffer  # TODO don't store the full tree
        else:
            stored_experience = list(
                itertools.islice(self.buffer, self.buffer_size - self.storage_size, self.buffer_size))
        pickle.dump(stored_experience, open(self.experience_path, 'wb'))

    @property
    def buffer_size(self):
        """
        :return: Current size of the buffer
        """
        return len(self.buffer)


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype = object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node
