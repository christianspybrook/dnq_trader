import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
import itertools
import argparse
import os
import pickle

from sklearn.preprocessing import StandardScaler


# import stock data:
def get_data():
	"""
	Return list of closing stock prices 
	"""

	# df = pd.read_csv('aapl_msi_sbux.csv')
	df = pd.read_csv('../data/apple_data.csv')

	return df.values




##########################################################
### Experience Replay Memory Class:                    ###
###     store and retrieve experiences                 ###
###                                                    ###
### Has constructor, store, and sample_batch functions ###
##########################################################

class ReplayBuffer:
	def __init__(self, obs_dim, act_dim, size):
		"""Initialize arrays buffers and pointers"""

		# initialize current state array, shape=(500, #_stocks * 2 + 1)
		self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
		# initialize next state array, shape=(500, #_stocks * 2 + 1)
		self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
		# initialize actions array, shape=(500, )
		# consists of int values: [0, #_stocks *2 + 1), as 8 bit representation
		self.acts_buf = np.zeros(size, dtype=np.uint8)
		# initialize rewards array, shape=(500, )
		self.rews_buf = np.zeros(size, dtype=np.float32)
		# initialize done flag array, shape=(500, )
		# consists of int values: [0, 1], as 8 bit representation
		self.done_buf = np.zeros(size, dtype=np.uint8)
		# instantiate pointer to start at 0
		self.ptr = 0
		# instantiate current buffer size to be 0
		self.size = 0
		# instantiate maximum buffer size
		self.max_size = size

	def store(self, obs, act, rew, next_obs, done):
		"""
		Store state, action, reward, next state, 
			 and done flag at pointer location (current step)
		"""

		# store currernt state
		self.obs1_buf[self.ptr] = obs
		# store next state
		self.obs2_buf[self.ptr] = next_obs
		# store actions
		self.acts_buf[self.ptr] = act
		# stores rewards
		self.rews_buf[self.ptr] = rew
		# stores done flag
		self.done_buf[self.ptr] = done
		# increment circular pointer for next call of store function
		self.ptr = (self.ptr + 1) % self.max_size
		# set current memory size (current size + 1) or (max size allowed)
		self.size = min(self.size + 1, self.max_size)

	def sample_batch(self, batch_size=32):
		"""
		Sample batch sized set of random indices 
			 (int values: [0, current buffer size]) 
			 for retrieving experiences
		"""

		idxs = np.random.randint(0, self.size, size=batch_size)

		# return dictionary of sampled transitions
		return dict(s=self.obs1_buf[idxs],
					s2=self.obs2_buf[idxs],
					a=self.acts_buf[idxs],
					r=self.rews_buf[idxs],
					d=self.done_buf[idxs])




def get_scaler(env):
	"""Return StandardScaler object to normalize state values"""
	### Note: replay buffer could be populated here, as well ###

	# instantiate list to store states encountered
	states = []

	# play one episode to get data for StandardScaler object
	### Note: run for multiple episodes for better accuracy ###
	for _ in range(env.n_step):
		# sample random action
		action = np.random.choice(env.action_space)
		# get data resulting from performing sampled action in environment
		state, reward, done, info = env.step(action)
		# store next state encountered
		states.append(state)
		# check for completion of episode
		if done:
			break

	scaler = StandardScaler()
	# fit StandardScaler object to states encountered
	scaler.fit(states)

	return scaler




def maybe_make_dir(directory):
	"""Utility to create directory, if it does not exist"""

	if not os.path.exists(directory):
		# create directory to store trained model and rewards encountered
		os.makedirs(directory)




class MLP(nn.Module):
	def __init__(self, n_inputs, n_action, n_hidden_layers=1, hidden_dim=32):
		super(MLP, self).__init__()

		# define input dimension
		M = n_inputs
		# instantiate list of hidden layers
		self.layers = []
		# define hidden layers
		for _ in range(n_hidden_layers):
			# define Linear layer
			layer = nn.Linear(M, hidden_dim)
			# set input dimension of next layer as output dimension of previous layer
			M = hidden_dim
			# add next layer
			self.layers.append(layer)
			# apply ReLU activation
			self.layers.append(nn.ReLU())

		# add final Linear layer
		self.layers.append(nn.Linear(M, n_action))
		# sequentially construct defined layers
		self.layers = nn.Sequential(*self.layers)

	def forward(self, X):
		return self.layers(X)

	def save_weights(self, path):
		torch.save(self.state_dict(), path)

	def load_weights(self, path):
		self.load_state_dict(torch.load(path))




def predict(model, np_states):
	"""Returns model predictions"""
	with torch.no_grad():
		# feed model input data
		inputs = torch.from_numpy(np_states.astype(np.float32))
		# collect output predictions
		output = model(inputs)

		return output.numpy()




def train_one_step(model, criterion, optimizer, inputs, targets):
	"""Runs on training step"""

	# convert data to tensors
	inputs = torch.from_numpy(inputs.astype(np.float32))
	targets = torch.from_numpy(targets.astype(np.float32))

	# zero parameter gradients
	optimizer.zero_grad()

	# perform forward pass
	outputs = model(inputs)
	loss = criterion(outputs, targets)
		
	# perform backward and optimize
	loss.backward()
	optimizer.step()




##############################################################################
### Stock Trading Environment Class:                                       ###
###     start episode, perform actions, and calculate results              ###
###                                                                        ###
### Has constructor, reset, step, _get_obs, _get_val, and _trade functions ###
##############################################################################

class MultiStockEnv:
	"""
	State: 
		vector, shape=(#_stocks * 2 + 1, ) ==>
		[#_shares stock 1, #_shares stock 2, ..., #_shares stock N, 
		 $_stock 1, $_stock 2, ..., $_stock N, cash on hand]

	Action: 
		categorical variable: has (#_actions^#_stocks) possibile values
		actions for each stock: 0 ==> sell, 1 ==> hold, 2 ==> buy
	"""

	def __init__(self, data, initial_investment=20000):
		# initialize 2-Dimensional data array of stock price time series
		self.stock_price_history = data
		# initialize number of steps (days) and number of different stocks
		self.n_step, self.n_stock = self.stock_price_history.shape

		### instance environment attributes ###

		# initialize starting cash on hand
		self.initial_investment = initial_investment
		# initialize step (day) counter
		self.cur_step = None
		# initialize number of stocks
		self.stock_owned = None
		# initialize stock prices
		self.stock_price = None
		# initialize trading cash
		self.cash_in_hand = None

		# initialize action space, shape=(#_actions^#_stocks, )
		#     consists of int values: [0, #_actions^#_stocks)
		self.action_space = np.arange(3**self.n_stock)

		# create action permutations: 
		#   nested list, shape=(#_actions^#_stocks, #_stocks) ==> 
		#   [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ..., [2,2,2]], 
		#   where   0 ==> sell
		#           1 ==> hold
		#           2 ==> buy
		self.action_list = list(
				map(
						list, itertools.product(
								[0, 1, 2], repeat=self.n_stock)
						)
				)

		# calculate size of state vector, shape=(#_stocks * 2 + 1, )
		self.state_dim = self.n_stock * 2 + 1

		# initialize reset function
		self.reset()


	def reset(self):
		"""Return state vector with starting state attributes"""

		# set pointer to first day of time series
		self.cur_step = 0
		# populate number of stocks array with zeros
		self.stock_owned = np.zeros(self.n_stock)
		# set current stock prices to values on first day of time series
		self.stock_price = self.stock_price_history[self.cur_step]
		# set trading cash to initial investment value
		self.cash_in_hand = self.initial_investment

		# return state vector
		return self._get_obs()


	def step(self, action):
		"""
		Return next state vector, reward, done flag, and 
		portfolio value dictionary, after performing action in environment
		"""

		# check that passed action is allowable
		assert action in self.action_space

		# get current portfolio value, before performing action
		prev_val = self._get_val()

		# increment current step (day) pointer
		self.cur_step += 1
		# update current stock prices attribute to next step (day)
		self.stock_price = self.stock_price_history[self.cur_step]

		# perform trading action on stocks
		self._trade(action)

		# get new portfolio value, after taking the action
		cur_val = self._get_val()

		# assign difference in porfolio values as reward
		reward = cur_val - prev_val

		# set done flag to True, if at end of time series
		done = self.cur_step == self.n_step - 1

		# store current value of portfolio in dictionary
		info = {'cur_val': cur_val}

		# return next state vector, reward, done flag, and 
		#   portfolio value dictionary, like OpenAI Gym API
		return self._get_obs(), reward, done, info


	def _get_obs(self):
		"""Return current state vector"""

		# initialize state vector, shape=(#_stock * 2 + 1, )
		obs = np.empty(self.state_dim)
		# populate number of each stock owned, shape=(3, )
		obs[:self.n_stock] = self.stock_owned
		# populate each current stock price, shape=(3, )
		obs[self.n_stock:2*self.n_stock] = self.stock_price
		# set last item to cash in hand, shape=(1, )
		obs[-1] = self.cash_in_hand

		return obs
		


	def _get_val(self):
		"""Return current portfolio value"""

		return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


	def _trade(self, action):
		"""Perform trading actions encoded in chosen action integer"""

		# get action vector, shape=(#_stocks, ) 
		#   for chosen action integer [0, #_actions^#_stocks)
		#   to map trading actions to each stock
		action_vec = self.action_list[action]

		# instantiate list to store index of stocks to sell
		sell_index = []
		# instantiate list to store index of stocks to buy
		buy_index = []

		# populate sell and buy lists
		for idx, val in enumerate(action_vec):
			# check for sell action
			if val == 0:
				# add stock index to sell list
				sell_index.append(idx)
			# check for buy action
			elif val == 2:
				# add stock index to buy list
				buy_index.append(idx)

		# first, sell all selected stocks
		if sell_index:
			# increment stocks to sell
			for idx in sell_index:
				# update new cash in hand value
				self.cash_in_hand += self.stock_price[idx] * self.stock_owned[idx]
				# remove all shares of stock sold
				self.stock_owned[idx] = 0

		# now, buy stocks, one by one, until not enough cash to continue buying
		if buy_index:
			can_buy = True
			while can_buy:
				# increment stocks to buy
				for idx in buy_index:
					# check that there is enough cash to buy
					if self.cash_in_hand > self.stock_price[idx]:
						# update new cash in hand value
						self.cash_in_hand -= self.stock_price[idx]
						# buy one share of stock
						self.stock_owned[idx] += 1
					else:
						can_buy = False




#######################################################################
### AI Agent Class:                                                 ###
###     given past experiences, learns to perform                   ###
###     actions to maximize future rewards                          ###
###                                                                 ###
### Has update_replay_memory, act, replay, load, and save functions ###
#######################################################################

class DQNAgent(object):
	def __init__(self, state_size, action_size):
		# instantiate size of state vector, shape=(#_stocks * 2 + 1, )
		# corresponds to number of inputs to neural network
		self.state_size = state_size
		# instantiate number of possible actions, shape=(#_actions^#_stocks)
		# corresponds to number of outputs of neural network
		self.action_size = action_size
		# initialize instance of Replay Memory object and set maximum size
		self.memory = ReplayBuffer(state_size, action_size, size=500)

		### set hyperparameters ###

		# set discount rate
		self.gamma = 0.95
		# set initial exploration rate
		self.epsilon = 1.0
		# set final exploration rate
		self.epsilon_min = 0.01
		# set factor by which to change exploration rate for each round
		self.epsilon_decay = 0.995
		# initialize instance of neural network
		self.model = MLP(state_size, action_size)

		# Loss and optimizer
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters())


	def update_replay_memory(
		self, state, action, reward, next_state, done):
		"""Take result of performing action and store in replay buffer"""

		self.memory.store(state, action, reward, next_state, done)

	def act(self, state):
		"""Take state and chose action based on Epsilon-Greedy method"""

		# generate random float number [0, 1) and compare to exploration rate
		if np.random.rand() <= self.epsilon:

			# return random action if chosen number is below threshold
			return np.random.choice(self.action_size)

		# get all Q-values for input state from neural network, 
		#     shape=(batch_size, #_actions^#_stocks)
		act_values = predict(self.model, state)

		# return action that leads to maximum Q-value
		return np.argmax(act_values[0])


	def replay(self, batch_size=32):
		"""Get samples from reply memory Learn"""

		# check if replay buffer contains full batch of data
		if self.memory.size < batch_size:
			# if not enough data in replay memory, exit function
			return

		# get sample batch of data from replay memory as dictionary
		minibatch = self.memory.sample_batch(batch_size)

		### separate transition values by indexing replay buffer dictionary ###

		# get current states
		states = minibatch['s']
		# get actions performed
		actions = minibatch['a']
		# get rewards from taking actions
		rewards = minibatch['r']
		# get next states
		next_states = minibatch['s2']
		# get done flags
		done = minibatch['d']

		# calculate estimated target Q(s', a), shape=(batch_size, ): 
		#     y_hat = r + gamma * max Q(s', a') over all a'

		# if next state is terminal state, set target to be only reward
		target = rewards + (1 - done) * self.gamma * \
		np.amax(predict(self.model, next_states), axis=1)

		###############################################################################
		### If using Keras API, target usually must have same shape as predictions. ###
		### However, network only needs to be updated for actions actually          ###
		### taken. Setting target equal to prediction for all values achieves       ###
		### this. Then, targets only need to be changed for actions taken.          ###
		### prediction: y = Q(s,a)                                                  ###
		### For all other actions (not taken), target is prediction.                ###
		### This leaves errors of actions not taken equal to zero, and              ###
		### they will not influence one-step of gradient descent.                   ###
		###############################################################################

		# get predictions for all actions for each sample, 
		#     even if action was not taken, 
		#     shape=(batch_size, #_actions^#_stocks)
		target_full = predict(self.model, states)
		# replace values for actions taken (those having actual targets)
		#     by double indexing: 
		#     rows where replacements will be made ==> [0, 1, ..., batch_size]
		#     columns where replacements will be made ==> states[actions taken]
		target_full[np.arange(batch_size), actions] = target

		# run one training step, using input states and estimated targets
		train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

		# check that exploration rate is still above minimum threshold
		if self.epsilon > self.epsilon_min:
			# update exploration rate by decay factor to reduce it over time
			self.epsilon *= self.epsilon_decay


	def load(self, name):
		"""Load model weights for testing with different configurations"""
		self.model.load_weights(name)


	def save(self, name):
		"""Save model weights for each run during training"""
		self.model.save_weights(name)




def play_one_episode(agent, env, is_train):
	"""Return current portfolio value, after once through time series"""

	# reset environment to initial state
	state = env.reset()
	# normalize data, shape=(#_stock * 2 + 1, )
	state = scaler.transform([state])
	# initialize done flag
	done = False

	# iterate through data, until reaching terminal state (end of data)
	while not done:
		# get next action to take from agent object
		action = agent.act(state)
		# get values for next state by performing action in environment
		next_state, reward, done, info = env.step(action)
		# normalize next state
		next_state = scaler.transform([next_state])
		# check if operating in train mode
		if is_train == 'train':
			# add most recent transition to replay buffer
			agent.update_replay_memory(state, action, reward, next_state, done)
			# run one step of gradient descent
			agent.replay(batch_size)
		# set state vairable to next state for next iteration
		state = next_state

	# return current portfolio value
	return info['cur_val']




# main function
if __name__ == '__main__':

	### set configuration variables ###

	# define path to save models
	models_folder = 'pt_trader_models'
	# define path to save rewards
	rewards_folder = 'pt_trader_rewards'
	# specify number of episodes to run
	num_episodes = 2000
	# specify batch size to sample from replay memory
	batch_size = 32
	# initialize amount of money in hand to begin trading
	initial_investment = 20000


	# instantiate parser object to run with Command Line arguments
	parser = argparse.ArgumentParser()
	# create mode argument to call either train, test, or random mode
	parser.add_argument('-m', '--mode', type=str, 
						required=True, help='either "train", "test", or "random"')
	args = parser.parse_args()

	# create models directory, if it does not exist
	maybe_make_dir(models_folder)
	# create rewards directory, if it does not exist
	maybe_make_dir(rewards_folder)

	# load in time series data of stock market
	data = get_data()
	# get number of time steps (days) and number of stocks
	n_timesteps, n_stocks = data.shape

	# define length of training data
	n_train = n_timesteps // 2

	# split data into train and test sets
	train_data = data[:n_train]
	test_data = data[n_train:]

	# instantiate environment object using training data and initial investment
	env = MultiStockEnv(train_data, initial_investment)
	# get state dimensionality, shape=(#_stocks * 2 + 1, )
	state_size = env.state_dim
	# get size of action space, (#_actions^#_stocks)
	action_size = len(env.action_space)
	# instantiate agent object
	agent = DQNAgent(state_size, action_size)
	# instantiate StandardScaler object
	scaler = get_scaler(env)

	# instantiate list to store final portfolio values at end of each episode
	portfolio_value = []

	# check if operating in test mode
	if args.mode == 'test':
		# load the proper StandardScaler object from training
		with open(f'{models_folder}/scaler.pkl', 'rb') as f:
			scaler = pickle.load(f)

		# recreate environment with test data
		env = MultiStockEnv(test_data, initial_investment)

		# Note: if epsilon = 0 ==> deterministic, no need to run multiple episodes
		# 		if epsilon = 1 ==> only random trades are made (default starting value)
		
		# make sure exploration rate is set to its minimum training value ==> 0.01
		agent.epsilon = agent.epsilon_min

		# load trained weights
		agent.load(f'{models_folder}/dqn.h5')

	# check if operating in random mode
	if args.mode == 'random':
		# recreate environment with test data
		env = MultiStockEnv(test_data, initial_investment)

		# neutralize factor that changes exploration rate for each round
		agent.epsilon_decay = 1.0

	# play episodes num_episodes times
	for episode in range(num_episodes):
		# grab current time to determine duration of each episode
		t0 = datetime.now()
		# get portfolio value by playing one episode
		val = play_one_episode(agent, env, args.mode)
		# get duration of episode
		dt = datetime.now() - t0

		# print portfolio value and episode duration after each episode
		print(f'Episode: {episode + 1}/{num_episodes}, Portfolio Value: {val:.2f}, Duration: {dt}')

		# add final portfolio value at end of each episode to list
		portfolio_value.append(val)

	# check if operating in train mode
	if args.mode == 'train':
		# save DQN model with trained weights
		agent.save(f'{models_folder}/dqn.h5')

		# save scaler used on training data
		with open(f'{models_folder}/scaler.pkl', 'wb') as f:
			pickle.dump(scaler, f)


	# save portfolio values (rewards) after each episode
	np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
