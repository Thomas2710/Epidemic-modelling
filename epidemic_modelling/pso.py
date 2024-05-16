from time import time
from random import Random
import inspyred
from inspyred.benchmarks import Benchmark
from inspyred import ec

class MySIRD(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.bounder = ec.Bounder([0.0] * self.dimensions, [1.0] * self.dimensions)
        self.maximize = False
        # self.global_optimum = [0 for _ in range(self.dimensions)]

    def generator(self, random, args):
        return [random.uniform(0.0, 1.0) for _ in range(self.dimensions)]

    def evaluator(self, candidates, args):
        # TODO: MSE and SIRD integration
        # fitness = []
        # for c in candidates:
        #     prod = 1
        #     for i, x in enumerate(c):
        #         prod *= math.cos(x / math.sqrt(i+1))
        #     fitness.append(1.0 / 4000.0 * sum([x**2 for x in c]) - prod + 1)
        # return fitness
        pass

def main(prng=None, display=False):
    if prng is None:
        prng = Random()
        print(prng)
        prng.seed(time()) 
    
    problem = MySIRD(3)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    ea.topology = inspyred.swarm.topologies.ring_topology
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator, 
                          pop_size=100,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          max_evaluations=30000, 
                          neighborhood_size=5)

    if display:
        best = max(final_pop) 
        print('Best Solution: \n{0}'.format(str(best)))
    return ea

if __name__ == '__main__':
    main(display=True)