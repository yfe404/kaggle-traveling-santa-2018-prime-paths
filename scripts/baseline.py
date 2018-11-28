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
    path = [0,]
    pbar = tqdm(total=len(ids))

    k = 0
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0) ## When a city is added to path, remove it from the list of remaining cities.
        xy = np.delete(xy, nearest_index, axis=0)
        pbar.update(1)
        if k % 1000 == 0: ## Run scoring every 1000 cities added to path
            scoring_thread = ScoringThread(target=utils.score_path, args=([path + list(ids) + [0]]))
            scoring_thread.start()

            backup_thread = BackupThread(target=utils.make_submission, args=("backup_baseline", path + list(ids) + [0]))
            backup_thread.start()
        k += 1

    path.append(0)

    return path



if __name__ == "__main__":
    path = solve()
    score = utils.score_path(path)

    print(score)

    util.make_submission('baseline_nearest_neighbour.{}'.format(np.round(score,2)), path)
