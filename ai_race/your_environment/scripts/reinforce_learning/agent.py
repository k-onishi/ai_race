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
            choices=[-0.5, 0.0, 0.5], height=240, width=320, channel=3, batch_size=32, gamma=0.999,
            epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, input_size=None,
            model_path=None, model_kwargs={}):
        model_kwargs["choices"] = choices
        model_kwargs["input_height"] = height
        model_kwargs["input_width"] = width
        model_kwargs["input_channels"] = channel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(**model_kwargs).to(self.device)
        self.teacher_model = model(**model_kwargs).to(self.device)
        self.teacher_model.load_state_dict(self.model.state_dict())
        self.teacher_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer = optim.RMSprop(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.experiences = ReplayMemory(max_experience)

        self.max_experience = max_experience
        self.actions = self.model.CHOICES
        self.height = height
        self.width = width
        self.channel = channel
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.num_output_random = 0
        self.num_output_inference = 0

        self.num_step = 0

        if input_size is None:
            self.input_size = self.model.input_size
        else:
            self.input_size = input_size

        self.model_path = model_path
        if self.model_path is not None:
            try:
                self.load(self.model_path)
                print("Load model from {}".format(self.model_path))
            except IOError:
                pass

    def preprocess(self, image):
        return torch.from_numpy(np.copy(image).astype(np.float32)).view(*self.input_size) / 256

    def policy(self, state):
        sample = random.random()
        threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1.0 * self.num_step / self.epsilon_decay)
        state = state.to(self.device)
        self.num_step += 1
        if sample > threshold:
            self.num_output_inference += 1
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
        else:
            self.num_output_random += 1
            return torch.tensor([[random.randrange(len(self.actions))]],
                    device=self.device, dtype=torch.long)

    def choice(self, action):
        return self.model.choice(action)

    def add_experience(self, *experience):
        self.experiences.push(*experience)

    def update(self):
        if len(self.experiences) < self.batch_size:
            print("Few experience: {} < {}".format(len(self.experiences), self.batch_size))
            return
        transitions = self.experiences.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                                device=self.device,
                                                dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_value = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.teacher_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_value, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_teacher(self):
        if self.model_path is not None:
            self.save(self.model_path)
        self.teacher_model.load_state_dict(self.model.state_dict())

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.update_teacher()

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
