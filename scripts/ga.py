import sys
sys.path.append('..')

from glob import glob

import random

from commons import utils
from commons.utils import ScoringThread, BackupThread

import numpy as np
import pandas as pd

from copy import deepcopy
from deap import base, creator, tools, algorithms

CITIES = pd.read_csv("../input/cities.csv")
IDS = CITIES.CityId.values[1:]
XY = np.array([CITIES.X.values, CITIES.Y.values]).T[1:]
POOL_DIR = "./genetic_pool/"



def generate_random_genome(proba=.5):

    path = None
    if np.random.random() < proba:
        print("Polling genome from genetic pool")
        sources = glob(POOL_DIR + "*")
        chosen_path = np.random.choice(sources)
        chosen_path = pd.read_csv(chosen_path)
        chosen_path = chosen_path['Path'].values

        path = list(chosen_path)
    else: 
        random_path = np.random.permutation(IDS)
        random_path = np.insert(random_path, 0, 0, axis=0)
        random_path = list(random_path)
        random_path.append(0)

        assert len(random_path) == len(CITIES) + 1
        assert (random_path[0] == 0)
        assert (random_path[-1] == 0)

        assert random_path[0] == random_path[-1] == 0
        assert len(set(random_path)) == len(random_path) - 1 == 197769

        path = random_path

    return path


#----------
# Observer registration
# Create a base class for observers
class Observer():
    def __init__(self):
        pass
    def notify(self, hof):
        print("New best individual found!")

        score = int(hof[0].fitness.values[0])

        pool_files = glob(POOL_DIR + "*")

        if score < min([int(x.split('/')[-1].split('.')[0]) for x in pool_files]): 
            backup_thread = BackupThread(target=utils.make_submission, args=(POOL_DIR, hof[0][0]))
            backup_thread.start()
            print("New best individual fitness score is {}".format(hof[0].fitness.values))
my_observer = Observer()
#----------

def _fitness(individual):
    _score = utils.score_path(individual[0])
    return (_score,)


def mutation_operator(individual, proba=.1):

    if np.random.random() < proba:

        for i in range(1, len(individual[0])-1):
            if random.random() < .01:
                j = np.random.randint(low=1, high=len(individual[0])-1, size=1)[0]

                temp = individual[0][i]
                individual[0][i] = individual[0][j]
                individual[0][j] = temp
        
    return (individual,)

def mate_operator(ind1, ind2):

    ind1[0] = np.array(ind1[0][1:-1]) -1
    ind2[0] = np.array(ind2[0][1:-1]) -1
    
    size = len(ind1[0])
    a, b = random.sample(range(size), 2)
          
    if a > b:
        a, b = b, a

    holes1, holes2 = [True]*size, [True]*size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[0][i]] = False
            holes2[ind1[0][i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1[0], ind2[0]
    k1 , k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[0][k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[0][k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[0][i], ind2[0][i] = ind2[0][i], ind1[0][i]


    ind1[0] = ind1[0] + 1
    ind1[0] = list(ind1[0])
    ind2[0] = ind2[0] + 1
    ind2[0] = list(ind2[0])
        
    ind1[0].insert(0, 0)
    ind1[0].append(0)

    ind2[0].insert(0, 0)
    ind2[0].append(0)

    return ind1, ind2


cxpb=0.08
mutpb=0.05
ngen=1000
pop_size=500

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


from multiprocessing import Pool
pool = Pool(processes=8)
toolbox = base.Toolbox()
toolbox.register("map", pool.map)

toolbox.register("generate_random_genome", generate_random_genome)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.generate_random_genome, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", _fitness)
toolbox.register("mate", mate_operator)
toolbox.register("mutate", mutation_operator, proba=mutpb)
toolbox.register("select", tools.selTournament, tournsize=3)

 
def main():

    pop = toolbox.population(n=pop_size)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    hof.register_observer(my_observer) 

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                        halloffame=hof)

    return pop, stats, hof



if __name__ == '__main__':
    print(CITIES.head())
    print(len(CITIES))
    print(IDS.shape)
    print(IDS[:10])

    print(XY.shape)
    print(XY[:10])

    pop, stats, hof = main()




