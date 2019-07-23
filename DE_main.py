"""
differential_evolution: The differential evolution global optimization algorithm
"""
#!/usr/bin/python
__author__ = "Arwa Shaker"


import numpy as np
import random
import matplotlib.pyplot as plt

def DE(fobj, bounds, popsize=10, its=10):
    dimensions = len(bounds)
    """----------------------------------------------------------------------------"""
    """                Step #1 generate the Initial Pop randomly                  """
    """----------------------------------------------------------------------------"""
    pop = np.random.rand(popsize, dimensions)

    """----------------------------------------------------------------------------"""
    """                Step #2 Evaluate the individuals fitness                    """
    """----------------------------------------------------------------------------"""
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    print(best_idx)
    best = pop_denorm[best_idx]

    """----------------------------------------------------------------------------"""
    """                Step #3 Differential Evolution Cycle                        """
    """----------------------------------------------------------------------------"""
    for i in range(its):
        print("Generation\t",i,"\n*****************************************\n",pop)
        for j in range(popsize):
            # set the target vector
            target = pop[j]
            """----------------------------------------------------------------------------"""
            """                     Setting Mutation Parameters                            """
            """----------------------------------------------------------------------------"""
            CRs = [0.1, 0.9, 0.2]
            Fmin = 0.6
            Fmax = 1.5
            MAXfit = 5000
            cfe = fobj.calls # number of calls to the fitness function

            F = ((Fmin-Fmax/MAXfit)*cfe) + Fmax
            """----------------------------------------------------------------------------"""
            """                     The Strategy #1 "rand/1/bin                            """
            """----------------------------------------------------------------------------"""
            # choose 3 vectors other than j
            idxs = [idx for idx in range(popsize) if idx != j]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            # mutation  clip is to keep the values to be between 0, and 1
            mutant_rand_1 = np.clip(r1 + F * (r2 - r3), 0, 1)
            # binomial  crossover
            CR = random.choice(CRs)
            cross_points = np.random.rand(dimensions) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial_1 = np.where(cross_points, mutant_rand_1, target)
            trial_1_denorm = min_b + trial_1 * diff
            f_1 = fobj(trial_1_denorm)
            """----------------------------------------------------------------------------"""
            """                     The Strategy #2 "rand/2/bin                            """
            """----------------------------------------------------------------------------"""

            # choose 5 vectors other than j
            idxs = [idx for idx in range(popsize) if idx != j]
            r1, r2, r3, r4,r5 = pop[np.random.choice(idxs, 5, replace=False)]
            Frand = np.random.random_sample()
            mutant_rand_2 = np.clip(r1 + Frand * (r2 - r3) + F*(r4 - r5), 0, 1)

            # binomial  crossover
            CR = random.choice(CRs)
            cross_points = np.random.rand(dimensions) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial_2 = np.where(cross_points, mutant_rand_2, target)  #pop[j] is the Target
            trial_2_denorm = min_b + trial_2 * diff
            f_2 = fobj(trial_2_denorm)

            """----------------------------------------------------------------------------"""
            """                 The Strategy #3 "current-to-rand/1                         """
            """----------------------------------------------------------------------------"""
            # choose 3 vectors other than j
            idxs = [idx for idx in range(popsize) if idx != j]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            rand = np.random.random_sample()
            trial_3 = np.clip(target + rand * (r1 - target) + F * (r2 - r3), 0, 1)
            # No crossover
            trial_3_denorm = min_b + trial_3 * diff
            f_3 = fobj(trial_3_denorm)

            """----------------------------------------------------------------------------"""
            """                 FIND THE BEST AMONG THE 3 TRAIL VECTORS                    """
            """----------------------------------------------------------------------------"""

            maximum = max(f_1,f_2,f_3)

            if f_1 == maximum and f_1 > fitness[j]:
                fitness[j] = f_1
                pop[j] = trial_1
                if f_1 > fitness[best_idx]:
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
@call_counter
def fobj(x):
    A = np.array([0.9, 0.1, 0.9, 0.1])
    value = 0.0
    for i in range(len(x)):
        value += x[i]*A[i]
    return value


"""----------------------------------------------------------------------------"""
"""                         Calling the DE and Plotting                        """
"""----------------------------------------------------------------------------"""
bounds=[(-1, 1)] * 4
it = list(DE(fobj,bounds))
x, f = zip(*it)

print("x",x[-1])
print("f",f[-1])
plt.plot(x, f, 'g^')

plt.show()


