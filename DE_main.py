"""
differential_evolution: The differential evolution global optimization algorithm
"""
#!/usr/bin/python
__author__ = "Arwa Shaker"


import numpy as np
import random
import matplotlib.pyplot as plt

def DE(objective_f, bounds, PS=10, its=100):
    dimensions = len(bounds)
    """----------------------------------------------------------------------------"""
    """                Step #1 generate the Initial Population randomly            """
    """----------------------------------------------------------------------------"""
    pop = np.random.rand(PS, dimensions)

    """----------------------------------------------------------------------------"""
    """                Step #2 Evaluate the individuals fitness                    """
    """----------------------------------------------------------------------------"""
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([objective_f(ind) for ind in pop_denorm])
    for ind in pop_denorm:
        print(ind, objective_f(ind))
    best_idx = np.argmax(fitness) #Returns the indices of the maximum values along an axis
    print(best_idx)
    best = pop_denorm[best_idx]
    print("Best individual\n============================================\n ",best)
    """----------------------------------------------------------------------------"""
    """                Step #3 Differential Evolution Cycle                        """
    """----------------------------------------------------------------------------"""
    for i in range(its):
        print("Generation\t",i,"\n*****************************************\n",pop)
        for j in range(PS):
            # set the target vector
            target = pop[j]
            """----------------------------------------------------------------------------"""
            """                     Setting Mutation Parameters                            """
            """----------------------------------------------------------------------------"""
            CRs = [0.1, 0.9, 0.2]  # from the used reference
            Fmin = 0.6 # from the used reference
            Fmax = 1.5 # from the used reference
            MAXfit = 5000 # from the used reference
            cfe = objective_f.calls # number of calls to the fitness function

            F = ((Fmin-Fmax/MAXfit)*cfe) + Fmax # from the ref a novel differential evolution mapping ---


            # This is the Ensemble of the mutation strategies

            """----------------------------------------------------------------------------"""
            """                     The Strategy #1 "rand/1/bin                            """
            """----------------------------------------------------------------------------"""

            # choose 3 vectors other than j
            idxs = [idx for idx in range(PS) if idx != j]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            # mutation  clip is to keep the values to be between 0, and 1
            mutant_rand_1 = np.clip(r1 + F * (r2 - r3), 0, 1)  #clipping the number to the interval, so values greater than 1 become 1

            # binomial  crossover
            CR = random.choice(CRs)
            cross_points = np.random.rand(dimensions) <= CR # finding the position where we should cross

            if not np.any(cross_points): # np.any -> Test whether any array element along a given axis evaluates to True
                cross_points[np.random.randint(0, dimensions)] = True  # if no cross points find jrand

            trial_1 = np.where(cross_points, mutant_rand_1, target) #Return elements chosen from mutant_rand_1or target depending on condition.
            trial_1_denorm = min_b + trial_1 * diff
            f_1 = objective_f(trial_1_denorm)
            """----------------------------------------------------------------------------"""
            """                     The Strategy #2 "rand/2/bin                            """
            """----------------------------------------------------------------------------"""

            # choose 5 vectors other than j
            idxs = [idx for idx in range(PS) if idx != j]
            r1, r2, r3, r4,r5 = pop[np.random.choice(idxs, 5, replace=False)]

            mutant_rand_2 = np.clip(r1 + F * (r2 - r3) + F*(r4 - r5), 0, 1)  # keep in [0,1]

            # binomial  crossover
            CR = random.choice(CRs) # randomly chosen from the variants
            cross_points = np.random.rand(dimensions) <= CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial_2 = np.where(cross_points, mutant_rand_2, target)  #pop[j] is the Target
            trial_2_denorm = min_b + trial_2 * diff
            f_2 = objective_f(trial_2_denorm)

            """----------------------------------------------------------------------------"""
            """                 The Strategy #3 "current-to-rand/1                         """
            """----------------------------------------------------------------------------"""
            # choose 3 vectors other than j
            idxs = [idx for idx in range(PS) if idx != j]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            rand = np.random.random_sample()
            trial_3 = np.clip(target + rand * (r1 - target) + F * (r2 - r3), 0, 1)
            # No crossover
            trial_3_denorm = min_b + trial_3 * diff
            f_3 = objective_f(trial_3_denorm)

            """----------------------------------------------------------------------------"""
            """                 FIND THE BEST AMONG THE 3 TRAIL VECTORS                    """
            """----------------------------------------------------------------------------"""

            maximum = max(f_1,f_2,f_3)

            if f_1 == maximum and f_1 > fitness[j]:  # fitness[j] = the objective of the target vector
                fitness[j] = f_1
                pop[j] = trial_1  # the trail 1 survives to the next generation
                if f_1 > fitness[best_idx]: # keep track of the best individuals
                    best_idx = j
                    best = trial_1_denorm

            if f_2 == maximum and f_2 > fitness[j]:
                fitness[j] = f_2
                pop[j] = trial_2
                if f_2 > fitness[best_idx]:
                    best_idx = j
                    best = trial_2_denorm

            if f_3 == maximum and f_3 > fitness[j]:
                fitness[j] = f_3
                pop[j] = trial_3
                if f_3 > fitness[best_idx]:
                    best_idx = j
                    best = trial_3_denorm

        yield best, fitness[best_idx]


# this code is to find the number of calls to the objective function

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper


"""----------------------------------------------------------------------------"""
"""                                The Fitness Function                        """
"""----------------------------------------------------------------------------"""
import pandas as pd
from List_Scheduling_Algo import List_Scheduling
@call_counter
def objective_f(x):
    X = pd.Series(x, index=['t1', 't2', 't3','t4','t5','t6'])
    Y = List_Scheduling(X)
    AssQuality_matrix = pd.DataFrame()
    for j in Y.index:
        AssQuality_matrix.loc[j, Y.loc[j]] = 0.5
    AssQuality_matrix = AssQuality_matrix.fillna(0.7)
    # creating the best solution
    AssQuality_matrix.loc['t1', 'w1'] = 1
    AssQuality_matrix.loc['t2', 'w2'] = 1
    AssQuality_matrix.loc['t3', 'w3'] = 1
    AssQuality_matrix.loc['t4', 'w1'] = 1
    AssQuality_matrix.loc['t5', 'w2'] = 1
    AssQuality_matrix.loc['t6', 'w3'] = 1
 
    value = 0.0
    for task in Y.index:
        if not pd.isna(Y[task]):
            value += AssQuality_matrix.loc[task, Y[task]]

    return value


"""----------------------------------------------------------------------------"""
"""                         Calling the DE and Plotting                        """
"""----------------------------------------------------------------------------"""
bounds=[(0, 1)] * 6
it = list(DE(objective_f,bounds))

x, f = zip(*it)

print("final solution**********************************************************\n",x[-1])
print("f",f[-1])
plt.plot(x, f, 'g^')

plt.show()


