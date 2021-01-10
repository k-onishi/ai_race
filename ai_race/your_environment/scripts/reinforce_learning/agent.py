import math
import random
from collections import namedtuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


Transition = namedtuple('Transition',
        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetworkAgent(object):
    def __init__(self, model, lr=1e-2, max_experience=10000,
            height=240, width=320, channel=3, batch_size=32, gamma=0.999,
            epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model().to(self.device)
        self.teacher_model = model().to(self.device)
        self.teacher_model.load_state_dict(self.model.state_dict())
        self.teacher_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.RMSprop(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.experiences = ReplayMemory(max_experience)

        self.max_experience = max_experience
        self.actions = model.CHOICES
        self.height = height
        self.width = width
        self.channel = channel
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.num_step = 0

    def preprocess(self, image):
        return torch.from_numpy(np.copy(image).astype(np.float32)).to(self.device).view(*self.model.input_size) / 256

    def policy(self, state):
        sample = random.random()
        threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1.0 * self.num_step / self.epsilon_decay)
        state = state.to(self.device)
        self.num_step += 1
        if sample > threshold:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.actions))]],
                    device=self.device, dtype=torch.long)

    def choice(self, action):
        return self.model.choice(action)

    def add_experience(self, *experience):
        self.experiences.push(*experience)

    def update(self):
        if len(self.experiences) < self.batch_size:
            print("experience: {}".format(len(self.experiences)))
            return
        print("update")
        transitions = self.experiences.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                                device=self.device,
                                                dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_value = self.model(state_batch).gather(1, action_batch)

        print("next_state_values")
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        print("next_state_values[non_final_mask]")
        next_state_values[non_final_mask] = self.teacher_model(non_final_next_states).max(1)[0].detach()
        print("expected_state_action_values")
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        print("loss")
        loss = F.smooth_l1_loss(state_action_value, expected_state_action_values.unsqueeze(1))

        print("optimizer")
        self.optimizer.zero_grad()
        print("before backward")
        loss.backward()
        print("clamp")
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        print("optimizer step")
        self.optimizer.step()
        print("update done")

    def update_teacher(self):
        self.teacher_model.load_state_dict(self.model.state_dict())
