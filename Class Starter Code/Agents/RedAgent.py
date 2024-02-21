from CybORG.Agents import BaseAgent
import random

class RedAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        # CSE233 Project: you should load your red agent model here

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        return random.randint(0, action_space - 1) # CSE233 Project: you should modify this line to get action from red agent
    
    def train(self): # CSE233 Project: you should modify this line to implement red agent training
        """allows an agent to learn a policy"""
        raise NotImplementedError # CSE233 Project: you should modify this line to implement red agent training


