import numpy as np
import scipy as sp
import random as rand
np.set_printoptions(suppress=True)

def selection(fitness, n:'number of pair of parents required'):
    # calculate probability of each p being selected as a parent
    f_inv = 1/(fitness/fitness.sum())
    prob = f_inv/f_inv.sum()
    # selecting n set of parents
    indexes = np.argsort(prob)[::-1]
    prob = np.sort(prob)[::-1]
    parents = []
    position = len(fitness)
    for i in range(n):
        parent = []
        for i in range(2):
            r = rand.random()
            for j in range(position):
                if r < prob[j]:
                    parent.append(indexes[j])
                    break
                else:
                    r -= prob[j]
        parents.append(parent)
    return parents
        
def crossover(p_set:'set of potential solutions', parents:'list of parent pairs indices'):
    # keep bits that are similar and perform uniform crossover for the rest
    n = len(parents)
    d = len(p_set[0])
    new_p_set = np.copy(p_set)
    for i in range(n):
        parent_A = p_set[parents[i][0]]
        parent_B = p_set[parents[i][1]]
        
        # go over each dimension and crossover if neccessary
        child_i = []
        for j in range(d):
            A_j = parent_A[j]
            B_j = parent_B[j]
            
            if abs(A_j - B_j) < 0.05:
                child_i.append((A_j + B_j)/2)
            elif rand.random() < 0.5:
                child_i.append(A_j)
            else:
                child_i.append(B_j)
        new_p_set = np.vstack([new_p_set, child_i])
    return new_p_set

def mutation(p_set:'set of positions', pm:'probability of mutation'):
    n = len(p_set)
    d = len(p_set[0])
    for i in range(n):
        for j in range(d):
            if rand.random() < pm:
                p_set[i][j] = (0.5-rand.random())*20
    return p_set

def survivor(p_set:'position set', n:'number of survivors', fitness:'fitness of positions'):
    indexes = np.argsort(fitness)[:10]
    fitness = np.sort(fitness)[:10]
    return p_set[indexes]
    
def genetic_optimize(f:'function', d:'dimension of function input', n:'starting positions', max_iter):
    # genetic algorithm hyperparameters
    # pc: probability of crossover, pm: probability of mutation
    pc = 0.5
    pm = 0.5
    
    # initialize and evaluate n potential solutions P(t)
    # limit the search space to +- 10 for each dimension
    
    p_set = (0.5-np.random.rand(1,d))*20
    for i in range(n-1):
        p = (0.5-np.random.rand(1,d))*20
        p_set = np.vstack([p_set, p])
        
    terminate = False
    i = 0
    i_no_improvement = 0
    while not terminate:
        # Evaluate current fitness
        fitness = np.apply_along_axis(f, 1, p_set)
        pop_fitness = fitness.mean()
        
        # Recombine P(t) to form C(t)

        # parent selection
        parents = selection(fitness, round(pc*n))
        # crossover
        p_set = crossover(p_set, parents)
        
        # mutation
        p_set = np.vstack([p_set,mutation(p_set[:n,:],pm)])
        
        # survivor selection
        fitness = np.apply_along_axis(f, 1, p_set)
        p_set = survivor(p_set, n, fitness)
        
        # check for termination criteria
        if fitness.mean() >= pop_fitness:
            i_no_improvement += 1
        else:
            i_no_improvement = 0
                          
        if i > max_iter:
            terminate = True
        elif i_no_improvement > 3:
            terminate = True
                          
        i += 1
        
    fitness = np.apply_along_axis(f, 1, p_set)
    indexes = np.argsort(fitness)
    return p_set[indexes[0]]