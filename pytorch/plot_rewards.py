import matplotlib.pyplot as plt
import numpy as np
import argparse

# instantiate parser object to run with Command Line arguments
parser = argparse.ArgumentParser()
# create mode argument to call either train or test mode
parser.add_argument('-m', '--mode', type=str, required=True,
										help='either "train" or "test"')
args = parser.parse_args()

### plot either training or test rewards ###

# check if operating in train mode
if args.mode == 'train':
	# load up rewards
	rewards = np.load(f'pt_trader_rewards/{args.mode}.npy')

	# plot progress of training rewards
	plt.title('Training Rewards History')

	plt.plot(rewards)

	plt.xlabel('Episodes')
	plt.ylabel('Portfolio Value')
else:
	# load up rewards
	rewards = np.load(f'pt_trader_rewards/test.npy')
	random_rewards = np.load(f'pt_trader_rewards/random.npy')

	# plot histogram of test rewards
	plt.title('RL Test vs Random Rewards Distribution')

	# plot rewards and capture bin heights
	m, _, __ = plt.hist(rewards, histtype='step', 
						bins=20, label='RL Trader', color='b')
	n, _, __ = plt.hist(random_rewards, histtype='step', 
						bins=20, label='Random Trades', color='r')

	# get height of largest bin
	v_ht = max(int(m.max()), int(n.max()))

	# plot initial investment value
	plt.vlines(20000, 0, v_ht, color='k', label='Start Value')

	# get extreme and average portfolio values
	test_min = rewards.min()
	rand_min = random_rewards.min()
	test_ave = rewards.mean()
	rand_ave = random_rewards.mean()
	test_max = rewards.max()
	rand_max = random_rewards.max()

	print(f'Trader: Ave = {test_ave:,.2f}, Min = {test_min:,.2f}, Max = {test_max:,.2f}')
	print(f'Random: Ave = {rand_ave:,.2f}, Min = {rand_min:,.2f}, Max = {rand_max:,.2f}')

	# plot minimum portfolio values
	plt.vlines(test_min, 0, v_ht / 4, color='b', label='RL Min/Max')
	plt.vlines(rand_min, 0, v_ht / 4, color='r', label='Random Min/Max')

	# plot average portfolio values
	plt.vlines(test_ave, 0, v_ht, color='b')
	plt.vlines(rand_ave, 0, v_ht, color='r')

	# plot maximum portfolio values
	plt.vlines(test_max, 0, v_ht / 4, color='b')
	plt.vlines(rand_max, 0, v_ht / 4, color='r')

	plt.xlabel('Portfolio Values')
	plt.legend()

plt.savefig(f'plots/{args.mode}.png')

plt.show() 
