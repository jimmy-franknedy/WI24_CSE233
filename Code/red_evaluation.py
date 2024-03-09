# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import time, os, torch
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.MainAgent import MainAgent
from Agents.RedAgent import RedAgent
import random
import numpy as np

MAX_EPS = 1
agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    
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

    # Path to directory to store red agent's optimal policy
    red_optimal_path = "red_optimal"

    # Add torch.load() code here
    red_agent.actor.load_state_dict(torch.load(os.path.join(red_optimal_path, "actor_policy.pth")))
    red_agent.critic.load_state_dict(torch.load(os.path.join(red_optimal_path, "critic_policy.pth")))

    # Intialize game and red agent observation
    num_steps = 30
    observation = env.reset()

    # Default action selection - random choice
    # action_space = wrapped_cyborg.get_action_space(agent_name)
    action_space = env.get_action_space(agent_name)

    total_reward = []
    actions = []
    for i in range(MAX_EPS):
        r = []
        a = []
        for j in range(num_steps):

            # Create the time bit vector
            time_vector = [0] * num_steps

            # Set current timestep
            time_vector[j] = 1

            # Combine the observation vector with time vector
            complete_observation = np.concatenate((observation, time_vector))

            # Get agent action
            _, _, action = red_agent.get_action(complete_observation)

            observation, rew, done, info = env.step(action)
            r.append(rew)
            print(j, ": ",rew)
            a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
        total_reward.append(sum(r))
        actions.append(a)
        observation = env.reset()