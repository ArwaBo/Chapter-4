from platypus import NSGAII, Problem, Real,Binary


class Belegundu(Problem):

    def __init__(self):
        super(Belegundu, self).__init__(2, 2, 2)
        self.types[:] = [Binary, Binary]
        self.constraints[:] = "<=0"

    def evaluate(self, solution):
        x = solution.variables[0]

        solution.objectives[:] = [-2 * x , 2 * x ]
        solution.constraints[:] = [1,  7]


algorithm = NSGAII(Belegundu())
algorithm.run(10000)
