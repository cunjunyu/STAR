import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant
import numpy as np


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        all_trajs = []
        all_teams = []
        for i in range(last_default_index):
            event = data_frame['events'][i]
            self.event = Event(event)
            trajs, teams = self.event.get_traj()  # (N,15,11,2)  # (N,15,11,1)
            if len(trajs) > 0:
                all_trajs.append(trajs)
                all_teams.append(teams)
                # print(i,len(trajs))
        all_trajs = np.concatenate(all_trajs,axis=0)
        all_teams = np.concatenate(all_teams,axis=0)

        return all_trajs,all_teams



