# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara

import inspect
import time
import numpy as np

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.RedAgent import RedPPOAgent
import random, os, torch, shutil

s = 153
MAX_EPS = 100
agent_name = 'Red'
random.seed(s)
torch.manual_seed(s)
random.seed(s)
np.random.seed(s)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    num_steps = 30
    
    # Select agent directory from different agents
    redAgent_1 = 'All_RedAction'
    redAgent_2 = 'Opt_RedAction'
    redAgent_3 = 'Opt_RedAction_ForceSleep'
    folder = redAgent_2

    # Create the policy directory path
    agent_folder = os.path.join(os.getcwd(), "Models", folder)

    # Select agent policy
    load_checkpoint = None      # Default value is None; which chooses the last policy the agent has learned
                                # Change None to a policy number if wanting to test a specific policy
    intial_generation = None
    if(load_checkpoint is None):
        # Index past [0] because thats where the reward information is stored
        load_checkpoint = sorted(os.listdir(agent_folder), key=lambda x: (int(x), -float('inf')) if x.isdigit() else (float('inf'), x), reverse=True)[1]
        intial_generation = load_checkpoint
    if(load_checkpoint is not None):
        intial_generation = load_checkpoint

    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    # Load blue agent
    blue_agent = WrappedBlueAgent

    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    cyborg.set_seed(s)
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")
    env.set_seed(s)
    
    # Create the red agent
    red_agent = RedPPOAgent(env.observation_space.shape[0]+num_steps, restore = True, ckpt = os.path.join(os.path.join(os.getcwd(), "Models", folder),str(load_checkpoint)), agent_type = folder)

    # Print policy check
    print("Testing",folder,"agent initialized with Generation",intial_generation)
    testing_start = time.time()
    total_reward = []
    actions = []
    for i in range(MAX_EPS):
        r = []
        a = []
        observation = env.reset()
        for j in range(num_steps):

            # Create the time bit vector
            time_vector = [0] * num_steps

            # Set current timestep
            time_vector[j] = 1

            # Combine the observation vector with time vector
            observation = np.concatenate((observation, time_vector))
            action = red_agent.get_action(observation)
            observation, rew, done, info = env.step(action)

            r.append(rew)
            a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

        total_reward.append(sum(r))
        actions.append(a)
        observation = env.reset()
    print("Average Total Rewards: {}".format(sum(total_reward)/len(total_reward)))
    testing_end = time.time()
    print("Test time:",time.strftime("%H:%M:%S", time.gmtime(testing_end-testing_start)))