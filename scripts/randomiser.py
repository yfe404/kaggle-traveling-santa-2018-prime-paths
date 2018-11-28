import sys

sys.path.append('..')

from commons import utils
from commons.utils import ScoringThread, BackupThread

import pandas as pd
import numpy as np

from tqdm import tqdm

import datetime, threading, time

from copy import deepcopy

import datetime, threading, time


if __name__ == "__main__":

    genome = pd.read_csv("./genetic_pool/1517307.csv")
    genome = genome.Path.values
    best_score = utils.score_path(genome)
    for i in range(1,len(genome)-2):
        __genome = deepcopy(genome)
        #rnd = np.random.randint(low=1, high=len(genome)-2)
        delta_kappa_pi = np.random.randint(low=1, high=3, size=1) 
        rnd = [i, (i+delta_kappa_pi)%(len(genome)-2)]
        temp = __genome[rnd[0]]
        __genome[rnd[0]] = __genome[rnd[1]]
        __genome[rnd[1]] = temp
        __score = utils.score_path(__genome)
        print(__score)
        if __score < best_score:
            best_score = __score
            print("Found new best!")
            utils.make_submission("./genetic_pool/", __genome)
            #genome = __genome

