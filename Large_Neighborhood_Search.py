"""
The Large Neighborhood Search implementation using ALNS package available online in https://pypi.org/project/alns/
"""
__author__ = "Arwa Shaker"
from alns import ALNS, State
from alns.criteria import SimulatedAnnealing
import random
import numpy.random as rnd
import pandas as pd
import numpy as np


SEED = 9876


"""----------------------------------------------------------------------------"""
"""                       Solution representation                              """
"""----------------------------------------------------------------------------"""

class TopKState(State):
    """
    Solution class for the top k worker task assignment problem, Series (vector) of tasks as index and workers as values
    the assignment quality matrix
    """

    def __init__(self, solution, AssQuality_matrix):
        self.solution = solution  # vector of tasks as indcies and workers as values
        self.tasks = solution.index
        self.workers = [self.solution[task] for task in self.solution.index]
        self.L_dash = []  # to keep the removed workers list
        self.AssQuality_matrix = AssQuality_matrix  # the assignment values matrix A

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
        worker_lose = pd.Series()
        lose = pd.Series()
        for worker in self.workers:
            worker_lose[worker] = 0.0
            for task in self.tasks:
                if (self.solution[task]) == worker:
                    worker_lose[worker] += self.AssQuality_matrix.loc[task, worker]
            # total objective value (mins) the quality of the worker at hand
            lose.loc[worker] = self.objective() - (self.objective() - worker_lose[worker])
            lose = lose.sort_values(ascending=True)
        return list(lose.index)


"""----------------------------------------------------------------------------"""
"""                              Destroy method                                """
"""----------------------------------------------------------------------------"""

degree_of_destruction = random.random()  # Random float x, 0.0 <= x < 1.0

def workers_to_remove(state):
    # How many worker to be removed based on the degree of destruction
    return int(len(state.workers) * degree_of_destruction)


def worst_removal(current, random_state):
    """
    Worst removal iteratively removes the 'worst' workers, that is,
    those workers that have the lowest quality.
    """

    destroyed = current.copy()

    worst_workers = destroyed.find_L()  # L

    h = workers_to_remove(current)  # h the number of workers to be removed
    p = 5  # the parameter p set to 5 according to ref ----

    while (h > 0):
        for task in destroyed.tasks:
            z = random.random()  # random number in [0,1)
            E = int((z ** p) * len(worst_workers))
            if destroyed.solution.loc[task] == worst_workers[E]:  # try to find the worst worker
                destroyed.solution.loc[task] = np.nan  # set the task with the worst worker to NAN
                destroyed.workers.remove(worst_workers[E])  # remove the worst worker from the solution
                destroyed.L_dash.append(worst_workers[E])
                h = h - 1
    return destroyed


"""----------------------------------------------------------------------------"""
"""                              Repair method                                 """
"""----------------------------------------------------------------------------"""


def greedy_repair(current, random_state):
    """
    Greedily repairs a solution,
    """
    # each worker has a capacity
    capacity = pd.Series()
    for worker in current.L_dash:
        capacity[worker] = 2

    current.L_dash = set(current.L_dash)  # L' the list of removed workers from Y'

    U_dash = []  # U' the list of unassigned task in  Y'

    for task in current.solution.index:
        if pd.isna(current.solution.loc[task]):
            U_dash.append(task)


    # the objective value of the destroyed solution
    objective_value_of_destroyed = current.objective()


    # find Delta fw,
    Delta_f = pd.DataFrame(index=[task for task in U_dash])

    for task in U_dash:
        for worker in current.L_dash:
            Delta_f.loc[task, worker] = (objective_value_of_destroyed + current.AssQuality_matrix.loc[
                task, worker]) - objective_value_of_destroyed


    for task in U_dash:
        if (capacity[Delta_f.loc[task, :].idxmax()]) > 0:
            current.solution.loc[task] = Delta_f.loc[task, :].idxmax()  # Get the BEST worker for the task at hand
            capacity[Delta_f.loc[task, :].idxmax()] -= 1  # reduce the capacity by one
            if (capacity[Delta_f.loc[task, :].idxmax()]) == 0:
                Delta_f.loc[:,Delta_f.loc[task, :].idxmax()] = 0.0  # Burn the Best worker (Best worker will not be chosen next time)

    return current


"""----------------------------------------------------------------------------"""
"""                              Initial solution                              """
"""----------------------------------------------------------------------------"""


Y = pd.Series(index=['t1','t2','t3','t4'])
Y.loc['t1'] = 'w1'
Y.loc['t2'] = 'w2'
Y.loc['t3'] = 'w3'
Y.loc['t4'] = 'w2'
AssQuality_matrix = pd.DataFrame()

for j in Y.index:
    AssQuality_matrix.loc[j,Y.loc[j]] = 0.5
AssQuality_matrix = AssQuality_matrix.fillna(1)
print("AssQuality_matrix\n",AssQuality_matrix)
print("Y\n",Y)


random_state = rnd.RandomState(SEED) #generating random numbers drawn from a variety of probability distributions
state = TopKState(Y, AssQuality_matrix)

initial_solution = greedy_repair(state, random_state)
print("########################initial###########################",type(initial_solution))
print("Initial solution objective is {0}.".format(initial_solution.objective()))


"""----------------------------------------------------------------------------"""
"""                              Heuristic solution                            """
"""----------------------------------------------------------------------------"""

alns = ALNS(random_state)
alns.add_destroy_operator(worst_removal)
alns.add_repair_operator(greedy_repair)
criterion = SimulatedAnnealing(1, 0.1, 0.6) #'start_temperature', 'end_temperature', and 'step'

result = alns.iterate(initial_solution, [3, 2, 1, 0.5], 0.8, criterion, iterations=100, collect_stats=True)

H_solution = result.best_state

objective = H_solution.objective()

print("########################the best heuristic solution########################\n",H_solution.solution, "with objective value\n", objective )