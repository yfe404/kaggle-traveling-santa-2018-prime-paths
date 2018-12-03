#!/usr/bin/env python
# coding: utf-8


from copy import copy
from math import hypot
from matplotlib import collections  as mc
from multiprocessing import Process
from subprocess import call
from sympy import isprime, primerange
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sympy
import uuid
from glob import glob



########################################## BEGIN CONSTANTS ##########################################
CITIES_FILE="../input/cities.csv"
PATH_TO_CROSSOVER_EXECUTABLE="../cpp/gsx2"
NB_PROCESSES=26
INDIV_SIZE=43
POPULATION_SIZE=100
MUTATION_RATE=0.1
SELECTIVE_PRESSURE=2
NB_GENERATIONS=50
GENETIC_POOL_PATTERN="../cpp/genetic_pool/*.tsp"
CITIES = pd.read_csv(CITIES_FILE, index_col=['CityId'])
########################################## END CONSTANTS ##########################################


########################################## BEGIN UTILS ##########################################
def write_to_file(seq, out):
    with open(out, "w") as f:
        f.write(" ".join([str(x) for x in seq]))

def read_from_file(inp):
    with open(inp, "r") as f:
        data = f.readlines()[0].strip().split(" ")
        return [int(x) for x in data]

class Node:
    """
    represents a node in a TSP tour
    """
    def __init__(self, num, coords):
        self.num = num # start position in a route's order
        self.x = coords[0]   # x coordinate
        self.y = coords[1]   # y coordinate

    def __str__(self):
        """
        returns the string representation of a Node
        """
        return self.num

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def euclidean_dist(self, other):
        """
        returns the Euclidean distance between this Node and other Node
        other - other node
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return hypot(dx, dy)


def read_from_julia(input_file):
    opts = []
    with open(input_file, "r") as f:
        size = int(f.readline())
        for i in range(size):
            distance,i,j = (x for x in f.readline().strip().split(" "))
            distance = float(distance)
            i = int(i)
            j = int(j)
            opts.append((distance,i,j))
    return opts

def plot_path(path, coordinates, k=0):
    # Plot tour
    lines = []
    distances = [] 
    for i in range(1, len(path)):
        line = [coordinates[path[i-1]], coordinates[path[i]]]
        (x1, y1), (x2, y2) = line
        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        distances.append(distance)   
        if distance > k:
            lines.append(line)
        
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
    return distances


class Tour(list):  # list with trailing 0
    penalized = ~CITIES.index.isin(sympy.primerange(0, len(CITIES)))

    @staticmethod
    def from_file(filename):  # from linkern's output or csv
        seq = [int(x) for x in open(filename).read().split()[1:]]
        return Tour(seq if seq[-1] == 0 else (seq + [0]))

    def score(self):
        df = CITIES.reindex(self)
        dist = np.hypot(df.X.diff(-1), df.Y.diff(-1))
        penalty = 0.1 * dist[9::10] * self.penalized[self[9::10]]
        _score = dist.sum() + penalty.sum()
        _score = np.round(_score,2)

        return _score

    def to_csv(self, filename):
        pd.DataFrame({'Path': self}).to_csv(filename, index=False)


########################################## END UTILS ##########################################



########################################## BEGIN GENETIC OPERATORS ##########################################

def crossover(a,b):
    write_to_file(a, "/tmp/aa.tsp")
    write_to_file(b, "/tmp/bb.tsp")
    
    call([PATH_TO_CROSSOVER_EXECUTABLE, "/tmp/aa.tsp", "/tmp/bb.tsp", "/tmp/out.tsp"])
    
    out_seq = read_from_file("/tmp/out.tsp")
    
    assert(len(out_seq)==len(a))
    assert(out_seq[0] == out_seq[-1] == 0)
    
    return out_seq


def mutate(indiv):
    
    i = np.random.randint(1, int(0.25*len(indiv.path)))
    ii = i+1
    
    j = np.random.randint(int(0.25*len(indiv.path))+1, int(0.50*len(indiv.path)))
    jj = j+1
    
    k = np.random.randint(int(0.50*len(indiv.path))+1, int(0.75*len(indiv.path)))
    kk = k+1
    
    l = np.random.randint(int(0.75*len(indiv.path))+1, len(indiv.path)-2)
    ll = l+1
    
    new_genome = indiv.path[0:i+1]

    new_genome.append(indiv.path[kk])
    new_genome.extend(indiv.path[kk+1:l+1])
    new_genome.append(indiv.path[jj])
    new_genome.extend(indiv.path[jj+1:k+1])
    new_genome.append(indiv.path[ii])
    new_genome.extend(indiv.path[ii+1:j+1])
    new_genome.append(indiv.path[ll])
    new_genome.extend(indiv.path[ll+1:])


    assert len(new_genome)==len(indiv.path)
    assert new_genome[0]==new_genome[-1]==0
    assert len(set(new_genome))==len(new_genome)-1
    
    return Individual(new_genome)


########################################## END GENETIC OPERATORS ##########################################




########################################## BEGIN MOCK JULIA ##########################################
def task_julia(start, end, path_to_input, path_to_output, k=20):
    call(["julia", "/home/yfe/kaggle-traveling-santa-2018-prime-paths/scripts_julia/ga_2opt.jl", str(start), str(end), "../input/cities.csv", path_to_input, path_to_output, str(k)])				
########################################## END MOCK JULIA ##########################################


########################################## BEGIN HEURISTICS ##########################################
def run_2opt(route):
    """
    improves an existing route using the 2-opt swap until no improved route is found
    best path found will differ depending of the start node of the list of nodes
        representing the input tour
    returns the best path found
    route - route to improve
    """
    improvement = True
    best_route = route
    best_distance = route.score()
    
    cities_per_thread = []
    if (len(route)-2)%NB_PROCESSES:
        cities_per_thread.append((len(route)-2)%NB_PROCESSES) 
    else:
        cities_per_thread.append(len(route)//NB_PROCESSES)
    for i in range(NB_PROCESSES-1):
        cities_per_thread.append(len(route)//NB_PROCESSES)

    
    buckets = [] 
    OPTS = []

    for i, bucket in enumerate(np.cumsum(cities_per_thread)):
        #print(i, bucket)
        if i == 0:
            buckets.append((1, bucket))#    population[0].path[1:-1][:bucket]     )

        else:
            buckets.append((buckets[-1][1], bucket)) #population[0].path[1:-1][np.cumsum(cities_per_thread)[i-1]:bucket]  )

    #print(buckets)    
   
    while improvement: 
        improvement = False
        
        processes = []
        file_out = ["/tmp/_raptor_julia_{}.tsp".format(x) for x in range(NB_PROCESSES)]
        file_in = "/tmp/input_julia.tsp"
        write_to_file(route, file_in)
        best_distance = route.score()
        
        
        for idx, bucket in enumerate(buckets):
            p = Process(target=task_julia, args=(bucket[0], bucket[1], file_in, file_out[idx]))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()

            
        for fic in file_out:
            opts = read_from_julia(fic)
            for opt in opts:
                OPTS.append(opt)
        #print("++++")
        #print(len(OPTS))
        #print("++++")
        
        if len(OPTS) > 0:
            improvement = True
            OPTS = sorted(OPTS, key=lambda x: x[0])
            #print( OPTS[0] )
            if len(OPTS) > 4:
                chosen_swap = random.choice(OPTS[:4])
            else:
                chosen_swap = OPTS[0]
            route = swap_2opt(route, chosen_swap[1], chosen_swap[2])
            best_distance = route.score()
            #print(best_distance)

            OPTS = []
            
       
    assert len(best_route)==len(route)
    assert best_route[0]==best_route[-1]==0
    assert len(set(best_route))==len(best_route)-1
    
    return route

########################################## END HEURISTICS ##########################################


########################################## BEGIN HYBRID GENETIC ALGORITHM ##########################################

class Individual():
    def __init__(self, path):
        self.path = path
        self.fitness = Tour(path).score()
        self.ranking = 0

def generate_population():

    population = []
    files = glob(GENETIC_POOL_PATTERN)

    for i in range(POPULATION_SIZE):
        seq = read_from_file(random.choice(files))
        assert seq[0]==seq[-1]==0
        assert len(set(seq))==len(seq)-1

        population.append(Individual(seq))
        
    
    return population

def get_rank(pop_size, selection_pressure, position):
    rank = 2 - selection_pressure + 2*(selection_pressure-1)*((position-1)/(pop_size-1))
    return rank

########################################## END HYBRID GENETIC ALGORITHM ##########################################


if __name__ == "__main__":



    population = generate_population()

    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    for i in range(len(population)):
        population[i].rank = get_rank(POPULATION_SIZE, SELECTIVE_PRESSURE, i)

    rnd = uuid.uuid4()
    for i in range(NB_GENERATIONS):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        for j in range(len(population)):
            population[j].rank = get_rank(POPULATION_SIZE, SELECTIVE_PRESSURE, j)

        population = sorted(population, key=lambda x: x.rank, reverse=False)
        
        best_score = population[-1].fitness
        write_to_file(seq=population[-1].path, out="./{}_{}_{}.tsp".format(best_score, rnd, i))
        print("Gen {} - Best: {}".format(i, best_score))
    
    
        if np.random.random() < MUTATION_RATE:
            parent = population[-1] 
            child = mutate(parent)
        else:
            p1 = population[-1] 
            p2 = population[-2] 
        
            child = crossover(p1.path, p2.path)

        child = Individual(child)

        child.path = run_2opt(Tour(child.path))
        child.fitness = Tour(child.path).score()
    
        print(child.fitness)
        if child.fitness < population[0].fitness:
            population[0] = child
    
