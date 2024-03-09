from CybORG.Agents import BaseAgent
import random

# Custom packages
import numpy as np
from tqdm import tqdm
import time, os, shutil
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import MultivariateNormal
from red_network import FeedForwardNN

class RedAgent():

    # Initialize Red agent
    def __init__(self, env, custom_action_set = []) -> None:
        super().__init__()

        """
        Generate the red agent  

        Parameters:
            env: CybORG
            custom_action_set (list, optional): List of custom action enums for red agent to use
            Defaults to all 888 possible actions
        """

        # Initialize hyperparameters for PPO training
        self._init_hyperparameters()

        # Extract environment information
        self.env = env

        # Default observation vector
        # self.obs_dim = env.observation_space.shape[0]

        # Time vector + Default observation vector
        self.obs_dim = env.observation_space.shape[0] + self.max_timesteps_per_episode          # Note: Ignored the above because observation was already given to red agent in wrapper
                                                                                                #       Calling 'print(self.obs_dim)' prints out '40'
                                                                                                #       Added 30 for the 30-bit time vector
        self.act_dim = env.get_action_space()                                                   # Note: Default action space is of size '888' (i.e total action space is 888 possible actions)

        if(len(custom_action_set) > 0):
            print("Red agent using custom_action_set")
            self.act_dim = len(custom_action_set)

        self.custom_action_set = custom_action_set.copy()

        # ALG STEP 1
        # Intialize actor and critic network
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Define optimizer for actor network
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        # Define optimizer for critic network
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Variable for covariance matrix (0.5 for stdev arbitrarily)
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create covariance matrix for 'get_action'
        self.cov_mat = torch.diag(self.cov_var)

        # Optimal policy selctor: max avg rtg per episode
        self.best_batch_mean_rtg = float('-inf')

        # Optimal policy saver:
        self.best_game = 0

        # Option for time vector to be included with obs
        self.time_vector_flag = False
        if(self.obs_dim > env.observation_space.shape[0]):
            self.time_vector_flag = True

    # Default hyperparameters for PPO training
    def _init_hyperparameters(self):

        # Timesteps per batch
        # (i.e how many games to play before updating actor-critic)

        # Combinatorial logic
        # Given the length of the action space (e.g 888) and
        # the length of the sequence we want the agent to follow (e.g we
        # want the action to take 3 actions in sequence) we want to batch
        # 'length of action space' x 'prob of desired action to be taken'
        # to give the agent enough chances to behave accordingly

        # self.max_timesteps_per_batch = (self.act_dim ** self.action_sequence_length) * self.multiplier
        self.max_timesteps_per_batch = 1000

        # How many actions we want agent to take
        # in a row
        self.action_sequence_length = 3                 # Want the agent to chose sleep 3 times before B-lining for Op Server

        # Multiplier to give agent a chance to
        # take on the desired behavior
        self.multiplier = 2

        # Timesteps per episode
        # (i.e how many timesteps per game)
        self.max_timesteps_per_episode = 30

        # Discount factor
        self.gamma = 0.99                               

        # Number of updates per epoch
        self.updates_per_iteration = 5                  

        # Clip ratio (value rec. by paper)
        self.clip = 0.2

        # Learning rate
        self.lr = 0.005

    # Implementation of PPO-Clip learning algorithm
    def train(self, total_games):

        print("Training started!")

        # Update training to take place over total_games
        # (i.e how many games agent should play to train?)

        # Track number of played games
        current_game = 0

        # Path to directory to store red agent's policies
        red_policy_folder_name = "red_policy"

        # Path to directory to store red agent's optimal policy
        red_optimal_folder_name = "red_optimal"

        # Create a directory to store red agent's policies

        # Note: Code will delete entire 'red_policy_folder' and re-create new policies during training
        #       To keep the old policies; change 'red_policy_folder' to new folder name
        #       Doing so, new policies can be trained and old ones are saved

        # Directory organized as below
        # red_policy
        #  |
        #   - 0
        #     |
        #      - Policy (i.e. neural network weights for actor-critic) for red agent at Game 0
        #   - 1
        #     |
        #      - Policy for red agent at Game 1
        #  ...
        #  - total_games

        red_policy_path = os.path.join(os.getcwd(), red_policy_folder_name)
        if os.path.exists(red_policy_path):
            shutil.rmtree(red_policy_path)
        os.makedirs(red_policy_path)
        for g in range(total_games):
            g_folder = os.path.join(red_policy_path,str(g))
            os.makedirs(g_folder)

        # Get start time
        start = time.time()
        global_start = start

        # ALG STEP 2
        # while current_timestep < total_timesteps:
        while current_game < total_games:

            # ALG STEP 3
            # Collect a batch
            # (i.e how many games to play before updating the actor-critic network)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate the average RTG across the batch
            # Note: The policy with the highest average RTG is considered the 'optimal' policy

            # Reshape batch_rtgs into a shape where each row represents one game
            batch_rtgs_reshaped = batch_rtgs.view(self.max_timesteps_per_batch, self.max_timesteps_per_episode)
            # print(batch_rtgs_reshaped)

            # Compute the total expected reward along the rows to get the RTG for each game
            batch_total_rtg_per_game = torch.sum(batch_rtgs_reshaped, dim=1)
            # print(total_rtg_per_game)

            # Calculate the average RTG across the batch
            batch_mean_rtg = torch.mean(batch_total_rtg_per_game).item()
            # print(batch_mean_rtg)

            # Update the max average RTG and track episode
            if(batch_mean_rtg > self.best_batch_mean_rtg):
                self.best_batch_mean_rtg = batch_mean_rtg
                self.best_game = current_game

            # Calculate value of observation
            # Note: Use '_' to upack log probs
            V, _ = self.compute_value(batch_obs, batch_acts)

            # ALG STEP 5
            # Calculate advantage
            advantage = batch_rtgs - V.detach()         # Could move .detach() into compute_value()

            # Normalize advantages
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            # Epoch update
            for _ in range(self.updates_per_iteration):
                
                # Calculate current 'policy'
                # Note: Use '_' to unpack 'V' values and get log probs
                V, curr_log_probs = self.compute_value(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage

                # Calculate actor-critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate actor gradients
                self.actor_optim.zero_grad()

                # Perform backward propagation for actor network
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate critic gradients  
                self.critic_optim.zero_grad()    

                # Perform backward propagation for critic network   
                critic_loss.backward()    
                self.critic_optim.step()            # something about setting 'retain_graph = True' to backward?

            # Determine red agent policy path
            current_policy_path = os.path.join(red_policy_path,str(current_game))
            # print("current_policy_path: ", current_policy_path)

            # Save red agent policy (i.e save actor-critic networks)
            torch.save(self.actor.state_dict(), os.path.join(current_policy_path, "actor_policy.pth"))
            torch.save(self.critic.state_dict(), os.path.join(current_policy_path, "critic_policy.pth"))

            # Finshed game time
            current = time.time()
            elapsed = current-start
            start = current

            print('Game(', current_game, ') Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)), '\tbatch_mean_rtg: ', batch_mean_rtg)
            current_game +=1       

        # Upload red agent's optimal policy to 'red_optimal' directory
        red_optimal_path = os.path.join(os.getcwd(), red_optimal_folder_name)
        if os.path.exists(red_optimal_path):
            shutil.rmtree(red_optimal_path)
        shutil.copytree(os.path.join(red_policy_path,str(self.best_game)), red_optimal_path)

        # Generate file, titled with the best episode
        f = os.path.join(red_optimal_path, 'episode_info')

        # Update file with episode information
        with open(f, 'w') as file:
            file.write("best_game: " + str(self.best_game))
            file.write("\nbest_batch_mean_rtg: " + str(self.best_batch_mean_rtg))

        print("Training completed!")
        print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(time.time()-global_start)))

    # Function to collect data within one batch
    def rollout(self):
    
        # Batch data to be collected
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Trach number of timesteps in batch
        current_timestep_batch = 0

        # Run through batches (i.e how many games to play before updating actor-critic)
        # with tqdm(desc='Collecting Samples') as pbar:

        while current_timestep_batch < self.max_timesteps_per_batch:
                
            # Reset episode reward
            episode_reward = []

            # Reset agent's environment
            obs = self.env.reset()
        
            # Run through episode (i.e play a game with current actor-critic network)
            # (i.e we're playing through 30-time steps; 1 full game)
            for current_timestep_episode in range(self.max_timesteps_per_episode):

                # Add time-vector if implemented
                if(self.time_vector_flag):

                    # Create the time bit vector
                    time_vector = [0] * self.max_timesteps_per_episode

                    # Set current timestep
                    time_vector[current_timestep_episode] = 1

                    # Combine the observation vector with time vector
                    obs = np.concatenate((obs, time_vector))

                # Collect red agent observation
                batch_obs.append(obs)

                # Get an red agent action
                action, log_prob, best_action_index = self.get_action(obs)

                # FOR EVALUATE:
                # _, _, action = self.get_action(obs)

                # Get reward and new observation from red agent action
                obs, rew, done, _ = self.env.step(best_action_index)

                # Record reward, action, action's log prob
                episode_reward.append(rew)              # Plus 1 because timestep starts at 0
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            # Update progress bar
            # pbar.update(1)
            # pbar.refresh()

            # print("Finished a sub-game!")
            # print("current_timestep_batch: ", current_timestep_batch)

            # Collect episodic length and rewards
            batch_lens.append(current_timestep_episode + 1) 
            batch_rews.append(episode_reward) 

            # Increment timestep in given batch
            current_timestep_batch += 1

        # Convert batch_obs into a single NumPy array
        # Note: this line alleviates the following warning when ran
        #       Not sure if '.float32' or 'float64' will have an impact on agent's overall learning?
        #       Adding this line, saved 7-9 seconds per game during training
        #
        #       UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        #       Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        #
        batch_obs = np.array(batch_obs, dtype=np.float64)
        batch_acts = np.array(batch_acts, dtype=np.float64)
        batch_log_probs = np.array(batch_log_probs, dtype=np.float64)

        # Reshape data as tensors in shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        # Compute 'rewards-to-go' or 'the expected return given an observation at timestep (t)'
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return batch data, 'rewards-to-go', and batch length
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):

      # Purpose: Compute rewards-to-go (rtg) per episode per batch to return.
      # Output:  Output shape will be (num timesteps per episode)
      batch_rtgs = []

      # Iterate through each episode backwards to maintain same order in batch_rtgs
      for ep_rews in reversed(batch_rews):

        # Discounted reward
        discounted_reward = 0

        # Go through rewards
        for rew in reversed(ep_rews):

            # Update discounted reward
            discounted_reward = rew + discounted_reward * self.gamma

            # Track discounted reward
            batch_rtgs.insert(0, discounted_reward)

      # Convert 'rewards-to-go' to a tensor, before returing
      batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

      return batch_rtgs

    def compute_value(self, batch_obs, batch_acts):

        # Purpose: Query critic network for a value V for each obs in batch_obs.
        # Ouput:   Use squeeze() to change dimensionality of tensor
        #          (e.g. calling squeeze on [[1], [2], [3]] will return [1, 2, 3])
        V = self.critic(batch_obs).squeeze()

        # Calculate log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return value and log probs
        return V, log_probs

    # def get_action(self, observation, action_space):
    def get_action(self, obs):

        # Default return from CSE233 skeleton code
        # return random.randint(0, action_space - 1) 

        # Query actor network for a mean action
        mean = self.actor(obs)

        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from distribution
        # Note: 'action' is a np.array of size 14 with each value ranging from [-1 to 1].
        #       The value represent the actions sampled from the Multivariate Normal Distribution
        action = dist.sample()

        # Retrieve action's log prob
        log_prob = dist.log_prob(action)


        # Original return: Return action and action's log prob
        # return action.detach().numpy(), log_prob.detach(), best_action
        # Note: action and action's log prob are tensors with computation graphs
        #       Use .detach() to remove graph and convert to numpy array


        # Custom return: Return action, action's log prob, best_action
        # Retrieve a 'best' action (i.e a single action) from the 14 values?
        # In this method we choose the action (a.k.a the action to be taken by the red agent) (represented by its index) with the highest action value
        # If downsizing action pool; re-map actions!

        best_action_index = np.argmax(action.detach().numpy().squeeze())

        # Check values
        # print("action.detach().numpy().squeeze():", action.detach().numpy().squeeze())
        # print("best_action_index: ", best_action_index)

        # Return action prob, log prob, and best action_index
        return action.detach().numpy(), log_prob.detach(), best_action_index