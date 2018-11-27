from threading import Thread
import pandas as pd
import numpy as np
from sympy import isprime, primerange


class ScoringThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
            print(self._return)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class BackupThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return



def make_submission(name, path):
    """Prepare a path for submission. The path will be written to name.csv."""
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    score = int(score_path(path))
    pd.DataFrame({'Path': path}).to_csv('{}{}.csv'.format(name, score), index=False)


def score_path(path):
    """Scoring function. Return the score of a given path as computed by Kaggle."""
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769

    cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
    pnums = [i for i in primerange(0, 197770)]
    path_df = cities.reindex(path).reset_index()
    
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))

    score = path_df.step_adj.sum()

    return score

