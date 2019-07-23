"""
The List-Scheduling algorithm:  map the solution vectors from continues to discrete ones
"""
#!/usr/bin/python
__author__ = "Arwa Shaker"

import numpy as np
import pandas as pd
def List_Scheduling(X):
    Y = pd.Series()
    L = X.sort_values(ascending=False)  # sorting tasks in X in  decreasing order
    W = pd.read_csv("C:/Users/Arwa/Desktop/datasets/MOO/worker_capacity.csv", index_col=0)
    k = 6
    print(L)
    print(W.loc['w1','Capacity'])
    print(W.index)
    for task in L.index:
        for worker in W.index:

            k_dash = W.loc[worker ,'Capacity']
            print("worker", worker, k_dash)
            if k_dash <= k and k_dash > 0 :
                Y.loc[task] = worker
                W.at[worker, 'Capacity'] = k_dash - 1
                print("W.at[worker, 'Capacity']", W.at[worker, 'Capacity'])
                break
    print(Y)

    return Y





X = pd.Series([10, 20, 30, 40, 50,90], index =['t1', 't2', 't3', 't4', 't5','t6'])

# print the Series
print(X)
List_Scheduling(X)