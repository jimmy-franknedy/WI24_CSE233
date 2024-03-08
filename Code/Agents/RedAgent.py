from CybORG.Agents import BaseAgent
import random

# Custom packages
import numpy as np
import time, os, shutil
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import MultivariateNormal
from red_network import FeedForwardNN

class RedAgent():

    # Initialize Red agent
    def __init__(self, env) -> None:
        super().__init__()

        # Initialize hyperparameters for PPO training
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]                           # Change this to get shape of red agent's observation space
                                                                                # Note: Ignored the above because observation was already given to red agent in wrapper
                                                                                #       Calling 'print(self.obs_dim)' prints out '40'
        self.act_dim = 888                                                      # Note: Logic for code can be found in CybORG tutorials Section 3 (Actions)
                                                                                #       In short, red agent can only choose 14 actions total

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

    # Default hyperparameters for PPO training
    def _init_hyperparameters(self):

        # Timesteps per batch
        # (i.e how many games to play?)
        self.max_timesteps_per_batch = 300              # Update to Cage 2; what is a batch?

        # Timesteps per episode        
        self.max_timesteps_per_episode = 30             # KEEP CONSTANT

        # Discount factor
        self.gamma = 0.99                               

        # Number of updates per epoch
        self.updates_per_iteration = 5                  # 5 might be enough?

        # Clip ratio (value rec. by paper)
        self.clip = 0.2

        # Learning rate
        self.lr = 0.005

    # Implementation of PPO-Clip learning algorithm
    def train(self, total_games):

        print("Training started!")

        # Counter to track timesteps
        current_timestep = 0

        # Track number of played games
        current_game = 0

        # Path to directory to store red agent's policies
        red_policy_folder_name = "red_policy"

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
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Update current timestep with collected timesteps from batch
            current_timestep += np.sum(batch_lens)
            # print("current_timestep: ", current_timestep)

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

            print('Game(', current_game, ') Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
            current_game +=1

        print("Training completed!")
        print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(global_start-time.time())))

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
        while current_timestep_batch < self.max_timesteps_per_batch:

            # Reset episode reward
            episode_reward = []

            # Reset agent's environment
            obs = self.env.reset()
        
            # Run through episode (i.e play a game with current actor-critic network)
            # (i.e we're playing through 30-time steps; 1 full game)
            for current_timestep_episode in range(self.max_timesteps_per_episode):

                # Increment timestep in given batch
                current_timestep_batch += 1

                # Collect red agent observation
                batch_obs.append(obs)                   # Initially 'obs' is reset; but should change as timesteps pass!

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

            # print("finished an episode!")
            # print("current_timestep_batch: ", current_timestep_batch)

            # Collect episodic length and rewards
            batch_lens.append(current_timestep_episode + 1) 
            batch_rews.append(episode_reward) 


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