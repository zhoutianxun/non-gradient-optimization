import numpy as np

def initialize(n, bounds):
    # generates a np array of n rows of random values that lies within the range given by bounds array
    # bounds array is a list of tuples, each tuple indicate the (min value, max_value) for given dimension
    # length of bounds is assumed to define the number of dimensions required
    bounds_range = np.apply_along_axis(lambda x:x[1]-x[0], 1, np.array(bounds))
    bounds_min = np.apply_along_axis(lambda x:x[0], 1, np.array(bounds))
    p_set_norm = np.random.rand(1, len(bounds))
    
    for i in range(n-1):
        p = np.random.rand(1, len(bounds))*bounds_range+bounds_min
        p_set_norm = np.vstack([p_set_norm, p])
        
    p_set = p_set_norm*bounds_range+bounds_min    
    return p_set_norm, p_set

def select_vectors(p_set, r, exclude_i):
    # chooses r vectors from the p_set, excluding the one at index: exclude_i 
    n = len(p_set)    
    idexes = [idx for idx in range(n) if idx != exclude_i]
    return p_set[np.random.choice(idexes, 3, replace=False)]

def get_trial_vector(vectors, c):
    # calculate trial vector using the 3 vector provided
    if len(vectors) != 3:
        return
    return vectors[0] + c*(vectors[1] - vectors[2])

def crossover(parent, trial_vector, pcr):
    # if pcr = 1, means all choose from trial vector, if 0, means all choose from parent vector
    choose_trial = np.random.rand(1,len(trial_vector)) <= pcr
    return (choose_trial*trial_vector + (1-choose_trial)*parent)[0]
    
def differential_evolution(f, bounds, c=0.5, pcr=0.7, n=15, max_iter=1000):
    # finds minimal for function f subject to boundary given by bounds.    
    # hyperparamters:
    # n random starting positions
    # c is the weight to multiply for the vector differential
    # pcr is the rate of crossover
    
    # initialize n random starting positions
    p_set_norm, p_set = initialize(n, bounds)
    bounds_range = np.apply_along_axis(lambda x:x[1]-x[0], 1, np.array(bounds))
    bounds_min = np.apply_along_axis(lambda x:x[0], 1, np.array(bounds))
    
    # positions variables
    fitness = np.apply_along_axis(f, 1, p_set)
    mean_fitness = np.mean(fitness)
    best_fitness = np.min(fitness)
    best_position = p_set[np.argmin(fitness)]
    best_position_list = []
    
    # loop variables
    terminate = False
    itr = 0
    i_no_improvement = 0
    
    while not terminate:
        for i in range(n):
            # select 3 other vectors from the positions and compute trial vector
            trial_vector = get_trial_vector(select_vectors(p_set_norm, 3, i), c)
            
            # crossover
            offspring_norm = crossover(p_set_norm[i], trial_vector, pcr)
            offspring_norm = np.clip(offspring_norm, 0, 1)
            offspring = offspring_norm*bounds_range+bounds_min 
            
            # select fitter between offspring and parent
            # minimization problem
            if f(offspring) < f(p_set[i]):
                p_set[i] = offspring
                p_set_norm[i] = offspring_norm
                
                if f(offspring) < best_fitness:
                    best_fitness = f(offspring)
                    p_set[np.argmin(fitness)] = offspring
                    p_set_norm[np.argmin(fitness)] = offspring_norm
                    fitness[np.argmin(fitness)] = f(offspring)
                    best_position = offspring
    
        best_position_list.append(best_position)
        
        # check for termination criteria
        if np.mean(fitness) >= mean_fitness:
            i_no_improvement += 1
        else:
            mean_fitness = np.mean(fitness)
            i_no_improvement = 0
        
        if itr > max_iter:
            terminate = True
        elif i_no_improvement > 20:
            terminate = True
        
        itr +=1
    
    print('the best solution is {} and function value is {}'.format(best_position, best_fitness))
    return best_position_list