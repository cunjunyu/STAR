
from Game import Game
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')

args = parser.parse_args()

data_root = '/workspace/MetaTRA/datasets/nba/source/CLE'
data_target = '/workspace/MetaTRA/datasets/nba/source/CLE'
if not os.path.exists(data_target):
    os.mkdir(data_target)
json_list = os.listdir(data_root)
print(json_list)
all_trajs = []
all_teams = []
for file_name in json_list:
	if '.json' not in file_name:
		continue
	json_path = data_root + '/' + file_name
	game = Game(path_to_json=json_path)
	trajs,teams = game.read_json()
	trajs = np.unique(trajs,axis=0)
	teams = np.unique(teams,axis=0)
	print(trajs.shape)
	print(teams.shape)
	all_trajs.append(trajs)
	all_teams.append(teams)

all_trajs = np.concatenate(all_trajs,axis=0)
all_trajs = np.unique(all_trajs,axis=0)
all_teams = np.concatenate(all_teams,axis=0)
all_teams = np.unique(all_teams,axis=0)
print(len(all_trajs))
print(len(all_teams))
print('all_trajs num:',all_trajs.shape[0])
print('all_teams num:',all_teams.shape[0])

np.save(data_target+'/all_trajs.npy',all_trajs)
np.save(data_target+'/all_teams.npy',all_teams)

# index = list(range(len(all_trajs)))
# from random import shuffle
# shuffle(index)
# train_set = all_trajs[index[:37500]]
# test_set = all_trajs[index[37500:]]
# print('train num:',train_set.shape[0])
# print('test num:',test_set.shape[0])

# np.save(data_target+'/train.npy',train_set)
# np.save(data_target+'/test.npy',test_set)
