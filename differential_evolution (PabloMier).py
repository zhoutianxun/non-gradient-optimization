import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs
import matplotlib.pyplot as plt

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, init='latin', display=False):
    dimensions = len(bounds)
    
    # initialize positions
    if init == 'random':
        pop = np.random.rand(popsize, dimensions)
    else:
        space = Space([(0.,1.)]*dimensions)
        lhs = Lhs()
        pop = np.asarray(lhs.generate(space.dimensions, popsize))
    
    min_b, max_b = np.asarray(bounds).T
    diff = max_b - min_b
    pop_denorm = min_b + pop * diff
    
    # compute fitness
    fitness = np.apply_along_axis(fobj, 1, pop_denorm)
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    # main iteration
    for i in range(its):
        for j in range(popsize):
            # generate 3 other random vectors 
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            
            # compute trial vector and perform boundary check
            # perform cross over
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            trial = cross_points*mutant + (1-cross_points)*pop[j]
            trial_denorm = min_b + trial * diff
            
            # compute fitness of trial vector
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        if display:
            print('differential evolution step {}: f(x)= {}'.format(i, fitness[best_idx]))
        
        yield best, fitness[best_idx]

# Benchmark function, Rosenbrock, global minima at x = 1
def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_with_noise(x, mu=0, sigma=0.5):
    # assume normal distributed noise
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + np.random.normal(mu, sigma)

bounds = [(-10,10)]*5



for i in ['random', 'latin']:
    it = list(de(rosen, bounds, its= 100, init=i))
    x, f = zip(*it)
    plt.plot(f, label='init={}'.format(i))
plt.legend()