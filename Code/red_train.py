# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import time

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.RedAgent import RedAgent
import random

MAX_EPS = 100
agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    

    
    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    # Load blue agent
    blue_agent = WrappedBlueAgent
    red_agent = RedAgent()
    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")

    max_episodes = 1
    max_timesteps = 1
    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()
        time_step = 0
        action_space = env.get_action_space('Red')
        for t in range(max_timesteps):
            time_step += 1
            action =  red_agent.get_action(observation, action_space) 
            observation, reward, done, _ = env.step(action)

            '''
            CSE233 Project: Here you should call red agent training function 
            '''
            # red_agent.train(...) # CSE233 Project: uncoment when you implement red agent training
            

