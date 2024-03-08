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

    # Get start time
    start = time.time()

    # Basic load
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    
    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    # Load blue agent
    blue_agent = WrappedBlueAgent

    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")

    # Load red agent
    red_agent = RedAgent(env)

    # Get intialization time
    elapsed_time = time.time() - start
    print('Initialization time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Create the action space
    optimized_red_action = [0] # sleep
    optimized_red_action += [1, 2, 3] # Discover Remote Services
    optimized_red_action += [i for i in range(4,17)] # Discover Network Services
    optimized_red_action += [i for i in range(17,30)] # Exploit Remote Service
    optimized_red_action += [i for i in range(758,771)] # Privilege Escalate
    optimized_red_action += [i for i in range(771,784)] # Impact

    # Train red agent (input: Number of games to train)
    red_agent.train(100)