
from collections import deque
import numpy as np

class ReplayBuffer():

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.states_curr = deque(maxlen=capacity)
        self.states_next = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.td_errors = deque(maxlen=capacity)

    def clear(self):
        self.states_curr.clear()
        self.states_next.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.td_errors.clear()

    def get_datalen(self):
        return len(self.states_curr)

    def get_avg_reward(self):
        return np.mean(self.rewards)

    def add_items(self, states, actions, rewards, dones, td_errors):
        assert len(states) >= 2
        states_curr = states[:-1]
        states_next = states[1:]
        assert len(states_curr) == len(states_next)
        assert len(states_curr) == len(actions)
        assert len(states_curr) == len(rewards)
        assert len(states_curr) == len(dones)
        assert len(states_curr) == len(td_errors)
        self.states_curr.extend(states_curr)
        self.states_next.extend(states_next)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.td_errors.extend(td_errors)

    def sample(self, batch_size=32, prob_delta_percentage=0.5):
        assert len(self.td_errors) == self.get_datalen(), "length: {} / {}".format(len(self.td_errors), self.get_datalen())
        abs_td_errors = np.abs(self.td_errors)
        ranking = np.argsort(abs_td_errors)
        # highest prob for highest error
        datalen = self.get_datalen()
        prob_avg = 1.0 / datalen
        prob_lowest = prob_avg * (1 - prob_delta_percentage)
        prob_highest = prob_avg * (1 + prob_delta_percentage)
        step = 2 * prob_avg * prob_delta_percentage / (datalen - 1)
        probs = prob_lowest + np.arange(datalen) * step
        # assert datalen == len(probs)
        elements_probs = probs[::-1][ranking]
        indices = np.random.choice(np.arange(datalen), size=(batch_size,), p=elements_probs).tolist()
        return self.choose_samples(indices)

    def sample_randomly(self, batch_size=32):
        datalen = self.get_datalen()
        # assert datalen >= batch_size
        indices = np.random.randint(0, datalen, size=(batch_size,), dtype=np.int32).tolist()
        return self.choose_samples(indices)

    def choose_samples(self, indices):
        states_curr = np.array([self.states_curr[i] for i in indices])
        states_next = np.array([self.states_next[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        return states_curr, states_next, actions, rewards, dones


if __name__ == "__main__":

    buffer = ReplayBuffer(80)
    td_errors = np.random.randint(100, size=(20,))
    buffer.states_curr.extend(td_errors)
    buffer.td_errors.extend(td_errors)
    print("td_errors:")
    print(td_errors)
    buffer.sample(10)
    exit()

    # test
    datalen = 20
    prob_delta_percentage = 0.5
    prob_avg = 1.0 / datalen
    prob_oldest = prob_avg * (1 - prob_delta_percentage)
    prob_newest = prob_avg * (1 + prob_delta_percentage)
    step = 2 * prob_avg * prob_delta_percentage / (datalen - 1)
    print(prob_oldest, prob_newest, step)
    probs = np.arange(prob_oldest, prob_newest, step)
    print("probs:", np.sum(probs), len(probs))
    print(probs)
    batch_size = 10
    indices = np.random.choice(np.arange(datalen), size=(batch_size,), p=probs).tolist()
    print("indices:")
    print(indices)


"""
python buffer.py
"""