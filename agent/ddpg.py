import torch
import torch.optim as optim
import numpy as np

from .model import ActorNN, CriticNN

import random
from collections import namedtuple, deque

import copy

class DDPGAgent:
    def __init__(self,config):
        state_size = config.state_size
        action_size = config.action_size
        seed = config.seed
        num_agents = config.num_agents

        self.device = config.device
        self.BATCH_SIZE = config.BATCH_SIZE
        self.TAU = config.TAU
        self.LR = config.LR
        self.BUFFER_SIZE = config.BUFFER_SIZE
        self.GAMMA = config.GAMMA

        # Actor-Network
        self.actor_local = ActorNN(state_size, action_size, seed).to(self.device)
        self.actor_target = ActorNN(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR)
        print("===================== Actor Network =========================")

        #Critic-Network
        self.critic_local = CriticNN(num_agents*state_size, num_agents*action_size , seed).to(self.device)
        self.critic_target = CriticNN(num_agents*state_size, num_agents*action_size , seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR)
        print("===================== Critic Network =========================")
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

         # Noise process
        self.noise = OUNoise(action_size, seed)

        self.t_step = 0
        
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state, add_noise=False, noise_decay=1.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noise_decay * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=config.BUFFER_SIZE)  
        self.batch_size = config.BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(config.seed)
        self.device = config.device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
