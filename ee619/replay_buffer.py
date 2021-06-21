import random
import numpy as np

class ReplayBuffer:
    def __init__(self, seed, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, minibatch_size):
        minibatch = random.sample(self.buffer, minibatch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*minibatch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
