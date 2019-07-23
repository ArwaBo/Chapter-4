"""
The multi-objective optimization functions
"""
#!/usr/bin/python
__author__ = "Arwa Shaker"
import pandas as pd

def MOO_weighted_Linear_sum(Y):
    A = pd.read_csv("C:/Users/Arwa/Desktop/datasets/MOO/A_matrix.csv", index_col=0)
    value = 0.0
    for task in Y.index:
        value +=  A.loc[task,Y[task]]

    print(value)
    return value




Y = pd.Series(['w1', 'w2', 'w1', 'w2', 'w3','w3'], index =['t1', 't2', 't3', 't4', 't5','t6'])

# print the Series
print(Y)

MOO_weighted_Linear_sum(Y)