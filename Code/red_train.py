# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara

import inspect
import time
import numpy as np
from tqdm import tqdm

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

    # Each game lasts 30 timesteps
    max_timesteps = 30  

    # Update 100 times (i.e create 100 generations) for training
    max_episodes = 100                                # DEFAULT: 100                  

    # Print 5 interval games for each generation
    # Should be a factor (e.g. 1/2 or 1/3 or 1/4 etc.) of number param for 'update_timestep'
    # -CALCULATION - 30 * number of games to play before printing
    print_interval = max_timesteps * 2500        # DEFAULT: 2 500 game print interval                    
    
    # Play & store (16^3 or 4096 rounded up) 5,000 games before updating the network
    # -CALCULATION - 30 * number of games to play before updating
    update_timestep = max_timesteps * 5000         # DEFAULT: 5 000 games before update
    # Variable tracking
    running_reward, time_step = 0, 0                # 'time_step' increments by taking a step in a game!
    set_of_actions = []

    # Training FLAGS                                # Default flag parameters are used to create 100 generation policies from scratch
    train_from_scratch = True                       # Set to 'False' if training from a certain generation
    train_from_checkpoint = not train_from_scratch  
    load_checkpoint = None                          # If loading from a checkpoint; this value should be the generation number      ** BUG HERE **

    # Make sure to set generation policy; if training from a specific checkpoint
    if(train_from_checkpoint is True and load_checkpoint is None):
        raise ValueError("Please indicate what checkpoint you'd like to load from!")

    # Make sure to reset load_checkpoint to None; if training from scratch
    if(train_from_scratch is True and load_checkpoint is not None):
        raise ValueError("Please set load_checkpoint to None if training from scratch!")

    # Select agent directory from different agents
    redAgent_1 = 'All_RedAction'
    redAgent_2 = 'Opt_RedAction'
    redAgent_3 = 'Opt_RedAction_ForceSleep'
    forceSleep = False

    #####################################################
    ############# Select agent for training #############

    folder = redAgent_2

    ############# Select agent for training #############
    #####################################################
    
    if(folder == 'Opt_RedAction_ForceSleep'):
        forceSleep = True

    # Create folder to store policies generated
    if(train_from_scratch):
        training_folder = os.path.join(os.getcwd(), "Models", folder)
        if os.path.exists(training_folder):
            shutil.rmtree(training_folder)
        os.makedirs(training_folder)

    # Create file to record generation and reward
    reward_file = os.path.join(os.getcwd(), "Models", folder, "Episode Rewards")
    if os.path.exists(reward_file):
        os.remove(reward_file)
    with open(reward_file, 'w') as file:
        file.write("Episode\tRewards\n")

    # Load scenario
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    # Load blue agent
    blue_agent = WrappedBlueAgent
    
    # Check initial generation
    intial_generation = 0
    if(load_checkpoint is not None):
        intial_generation = load_checkpoint
        load_checkpoint = os.path.join(os.path.join(os.getcwd(), "Models", folder),str(load_checkpoint))

    # Set up environment with blue agent running in the background and red agent as main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    cyborg.set_seed(s)
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")
    env.set_seed(s)

    # Create the red agent
    red_agent = RedPPOAgent(env.observation_space.shape[0]+max_timesteps, restore = train_from_checkpoint, ckpt = os.path.join(os.path.join(os.getcwd(), "Models", folder),str(load_checkpoint)), agent_type = folder)
    start = time.time()
    global_start = start

    print("\nTraining",folder,"agent initialized with Generation",intial_generation,"\n")
    
    # Gather samples
    i_episode = 1

    # Initialize the progress bar outside of the main loop
    # NOTE: pbar COULD DISPLAY the INCORRECT amount of timesteps
    #       this is use a display issue; pbar does in fact go through everything!
    #       Use the print statements below to check if needed!
    pbar = tqdm(total=update_timestep, desc='Batching Progress', unit='time_step')

    # Creates relationship betweem number of times to update the network before finishing training
    while(i_episode < max_episodes+1):
        state = env.reset()
        set_of_actions.clear()
        game_reward = 0
        
        for t in range(max_timesteps):
            # Update timestep
            time_step += 1
            pbar.update(1)

            # Create the time bit vector
            time_vector = [0] * max_timesteps

            # Set current timestep
            time_vector[t] = 1

            # Combine the observation vector with time vector
            state = np.concatenate((state, time_vector))

            action = red_agent.get_action(state)
            set_of_actions.append(action)
            
            state, reward, done, _ = env.step(action)
            red_agent.store(reward, done)

            running_reward += reward
            game_reward += reward

            # Print interval to see reward progress
            if time_step % print_interval == 0:

                # Check pbar progress to confirm
                # print("\nprint - pbar proress value: ", pbar.n)

                running_reward = float((running_reward / time_step))
                print('\nGame {} Avg reward: {}'.format(int(time_step/max_timesteps), running_reward))
                print("Action set:", set_of_actions,"\n")
                running_reward = 0

            # Update interval for neural network
            if time_step % update_timestep == 0:

                # Check pbar progress to confirm
                # print("update - pbar proress value: ", pbar.n)

                # Time the sample colletion
                end = time.time()
                elapsed_time = end-start
                print("\nUpdating to Generation", i_episode+intial_generation, "\nTime to collect samples: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                # Increment to next generation
                i_episode += 1

                # Update neural network
                red_agent.train()

                # Save the neural network to it's generation file
                generation_file_name = os.path.join(training_folder, str(i_episode))
                torch.save(red_agent.policy.state_dict(),generation_file_name)

                # Reset variables
                red_agent.clear_memory()
                start = time.time()
                pbar.close()
                print("\n")
                pbar = tqdm(total=update_timestep, desc='Batching Progress', unit='time_step')

        # Record the generation and corresponding reward

        # Record the reward for the current game
        with open(reward_file, 'a') as rew_file:
            rew_file.write(f"{i_episode}\t{game_reward}\n")

        # Reset starting actions if agent is 'Opt_RedAction_ForceSleep'
        if(folder == 'Opt_RedAction_ForceSleep'):
            red_agent.reset_start_actions()

    global_end = time.time()
    print("\nTotal training time: ",time.strftime("%H:%M:%S", time.gmtime(global_end-global_start)))