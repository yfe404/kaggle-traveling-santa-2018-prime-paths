import sys

sys.path.append('..')

from commons import utils
from commons.utils import ScoringThread, BackupThread

import pandas as pd
import numpy as np

from tqdm import tqdm

import datetime, threading, time


import datetime, threading, time


        
def solve():

    cities = pd.read_csv("../input/cities.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = pd.read_csv("genetic_pool/1517028.csv")
    path = path.Path.values
    pbar = tqdm(total=500)

    current_score = utils.score_path(path)
    
    k = 0
    while True:
        for i in np.random.randint(low=0, high=len(path)-1, size=(500,)):
            last_x, last_y = cities.X[path[i]], cities.Y[path[i]]
            dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
            ten_nearest = np.argsort(dist)[:10]
            nearest_index = np.random.choice(ten_nearest, size=1)[0]
#            nearest_index = dist.argmin()

            nearest_neighbor = ids[nearest_index] ## new value at i+1
            actual_idx_for_nearest_neighbor = np.argwhere(path==nearest_neighbor)[0][0] # place for value at i+1

            temp = path[i+1]
            path[i+1] = nearest_neighbor
            
            path[actual_idx_for_nearest_neighbor] = temp

            new_score = utils.score_path(path)

            if new_score < current_score:
                print("Found new best score: {}".format(new_score))
                current_score = new_score
                utils.make_submission("./genetic_pool/", path)
            else :
                print("New score NOT better: {} -- keeping {}".format(new_score, current_score))
                path[actual_idx_for_nearest_neighbor] = path[i+1]
                path[i+1] = temp
                 
            pbar.update(1)
    return path



if __name__ == "__main__":
    path = solve()
    score = utils.score_path(path)

    print(score)

    util.make_submission('baseline_nearest_neighbour.{}'.format(np.round(score,2)), path)
