# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# only changes involve keeping track of decoys, adding scanning states, and reduction of action space

from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory
from CybORG.Agents import BaseAgent
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RedPPOAgent(BaseAgent):
    def __init__(self, input_dims, action_space=[], lr=0.001, betas=[0.9, 0.990], gamma=0.99, 
                 K_epochs=6, eps_clip=0.2, restore=False, ckpt=None, agent_type=None):
        super().__init__()

        # Used for 'get_action' logic given 'Opt_RedAction_ForceSleep' agent
        self.forceSleep = False

        # Intialize action set for specific agents
        action_space = []

        if(agent_type == 'All_RedAction'):

            # Entire action set for red agent
            action_space = [0]                              # Sleep
            action_space += [1, 2, 3]                       # Discover Remote Services
            action_space += [i for i in range(4,17)]        # Discover Network Services
            action_space += [i for i in range(17,30)]       # Exploit Remote Service
            action_space += [i for i in range(758,771)]     # Privilege Escalate
            action_space += [i for i in range(771,784)]     # Impact

        elif(agent_type == 'Opt_RedAction'):

            # Optimized action set for given seed
            action_space = [0, 3, 13, 26, 767, 6, 19, 760, 1, 7, 26, 761, 11, 24, 765, 778]

        elif(agent_type == 'Opt_RedAction_ForceSleep'):

            # Optimized action set for given seed, forcing to sleep intial 3 turns
            action_space = [3, 13, 26, 767, 6, 19, 760, 1, 7, 26, 761, 11, 24, 765, 778]
            self.forceSleep = True

        self.action_space = action_space
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = Memory() 
        self.policy = ActorCritic(input_dims, len(self.action_space)).to(device)
        self.start_actions = [0,0,0]

        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(input_dims, len(self.action_space)).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MSE_loss = nn.MSELoss()

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()

    def get_action(self, observation, deterministic=False):

        # Logic to sleep first 3 turns
        if(self.forceSleep):
            if len(self.start_actions) > 0:
                action = self.start_actions[0]
                self.start_actions = self.start_actions[1:]

        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)  
        action = self.old_policy.act(state, self.memory, deterministic=deterministic)
        return self.action_space[action]

    def train(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device).detach()
        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())

    def reset_start_actions(self):
        self.start_actions = [0,0,0]