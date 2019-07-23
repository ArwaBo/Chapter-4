"""
The Large Neighborhood Search implementation using ALNS package available online in https://pypi.org/project/alns/
"""

from alns import ALNS, State
from alns.criteria import HillClimbing
import math
import itertools
import random
import numpy.random as rnd
import pandas as pd
import networkx as nx
import numpy as np
import tsplib95
import tsplib95.distances as distances

import matplotlib.pyplot as plt

SEED = 9876


class TopKState(State):
    """
    Solution class for the top k worker task assignment problem, Series (vector) of tasks as index and workers as values
    the assignment quality matrix
    """

    def __init__(self, solution, AssQuality_matrix):
        self.solution = solution
        self.tasks = solution.index
        self.workers = [self.solution[task]for task in self.solution.index]
        self.AssQuality_matrix = AssQuality_matrix
        print(self.workers)

    def copy(self):
        """
        Helper method to ensure each solution state is immutable.
        """
        return TopKState(self.solution.copy(), self.AssQuality_matrix.copy())

    def objective(self):
        """
        The objective function is simply the sum of all Assignment qualities
        """
        value = 0.0
        for task in self.solution.index:
            if not pd.isna(self.solution[task]):
                value += self.AssQuality_matrix.loc[task, self.solution[task]]
        return value

    def find_L(self):
        benefit = pd.Series()
        for task in self.solution.index:
                benefit.loc[self.solution[task]] = self.objective() - self.AssQuality_matrix.loc[:, self.solution[task]].sum()
        benefit.sort_values(ascending=False)
        return list(benefit.index)


"""----------------------------------------------------------------------------"""
"""                              Destroy method                                """
"""----------------------------------------------------------------------------"""

degree_of_destruction = 0.9 #random.random()        # Random float x, 0.0 <= x < 1.0
print("degree_of_destruction",degree_of_destruction)
def workers_to_remove(state):
    print("workers_to_remove",int(len(state.workers) * degree_of_destruction))
    return int(len(state.workers) * degree_of_destruction)


def worst_removal(current, random_state, AssQuality_matrix):
    """
    Worst removal iteratively removes the 'worst' workers, that is,
    those workers that have the lowest quality.
    """
    destroyed = current.copy()

    worst_workers = destroyed.find_L()  # L
    print("worst_workers",worst_workers)

    h = workers_to_remove(current) # h the number of workers to be removed
    p = 5 # the parameter p set to 5 according to ref ----

    L_dash = []
    while (h>0):
        for task in destroyed.tasks:
            z = random.random()   # random number in [0,1)
            E = int((z**p)* len(worst_workers))
            print("worst_workers[-idx - 1]", worst_workers[E])
            if destroyed.solution.loc[task] == worst_workers[E]:
                destroyed.solution.loc[task] = np.nan
                destroyed.workers.remove(worst_workers[E])
                L_dash.append(worst_workers[E])
                h = h-1
    return destroyed, L_dash


"""----------------------------------------------------------------------------"""
"""                              Repair method                                 """
"""----------------------------------------------------------------------------"""


def greedy_repair(current, L_dash, capacity, AssQuality_matrix):
    """
    Greedily repairs a solution,
    """
    L_dash = set(L_dash)# L' the list of removed workers from Y'
    print("L_dash", L_dash)
    U_dash = [] # U' the list of unassigned task in  Y'
    for task in current.solution.index:
        print("current.solution.loc[task]",current.solution.loc[task])
        if  pd.isna(current.solution.loc[task]):
            U_dash.append(task)
    print("U_dash", U_dash)

    # find Delta fw,

    Delta_f =  pd.DataFrame( index=[task for task in U_dash])
    objective_value_of_destroyed = current.objective()
    print("objective_value_of_destroyed", objective_value_of_destroyed)




    for task in U_dash:
         for worker in L_dash:
             Delta_f.loc[task,worker] = (objective_value_of_destroyed + AssQuality_matrix.loc[task,worker]) - objective_value_of_destroyed

    print("Delta_f-----------------------------------------------------------\n", Delta_f)
    for task in U_dash:

        if (capacity[Delta_f.loc[task, :].idxmax()]) > 0:
            current.solution.loc[task] = Delta_f.loc[task,:].idxmax() # Get the BEST worker
            capacity[Delta_f.loc[task, :].idxmax()] -=1 # reduce the capacity by one
            #Delta_f.loc[task, Delta_f.loc[task,:].idxmax()] = 0.0  # Burn the Best worker (Best worker will not be chosen next time)


    print("Delta_f-----------------------------------------------------------\n", Delta_f)
    print("repaired sol\n", current.solution)
    print(capacity)
    return current

random_state = rnd.RandomState(SEED)
Y = pd.Series(index=['t1','t2','t3','t4'])
Y.loc['t1'] = 'w1'
Y.loc['t2'] = 'w1'
Y.loc['t3'] = 'w3'
Y.loc['t4'] = 'w2'
AssQuality_matrix = pd.DataFrame()

for j in Y.index:
    AssQuality_matrix.loc[j,Y.loc[j]] = 0.5
AssQuality_matrix = AssQuality_matrix.fillna(1)
print("AssQuality_matrix",AssQuality_matrix)
solution = TopKState(Y, AssQuality_matrix)


print(solution .objective())
print(solution .tasks)

destroyed, L_dash = worst_removal(solution, random_state, AssQuality_matrix)
print("destroyed\n",destroyed.solution)
capacity = pd.Series()
for worker in L_dash:
    capacity[worker] = 2
greedy_repair(destroyed, L_dash, capacity, AssQuality_matrix)