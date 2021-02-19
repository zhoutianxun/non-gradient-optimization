import numpy as np
from scipy.optimize import differential_evolution as scipy_de
from scipy.special import gammaln
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from skopt import gp_minimize

class _Optimizer:
    """
    Optimizer class
    
    Contains common attributes and methods used in all types of optimizers
    
    """
    
    def __init__(self, fun, bounds, popsize, maxiter, max_eval, disp):        
        self.fun = fun
        self.bounds = bounds
        self.popsize = popsize
        self.maxiter = maxiter
        self.max_eval = max_eval
        self.disp = disp
        
        self.solved = False
        self.converge_data = None
        self.x = None
        self.fx = None
    
    def result(self):
        """
        Returns
        -------
        tuple
            (optimal result found, functional value of optimal result)
        """
        if self.solved:
            return self.x, self.fx
        else:
            print("Please solve the optimization first by calling .solve() method")
    
    def convergence_plot(self):
        """
        Returns
        -------
        None
            plots the convergence profile over the course of the iterative optimization process
        """
        if self.solved:
            plt.plot(self.converge_data[0], self.converge_data[1])
            plt.show()
        else:
            print("Please solve the optimization first by calling .solve() method")
            
        
class DifferentialEvolution(_Optimizer):
    """
    DifferentialEvolution class
    
    The solver is built as a wrapper over scipy.optimize.differential_evolution
    """
    def __init__(self, fun, bounds, popsize=20, maxiter=5000, max_eval=10000, disp=False):
        super().__init__(fun, bounds, popsize, maxiter, max_eval, disp)
        self.max_eval = max_eval
        if max_eval//popsize -1 > maxiter:
            self.maxiter = maxiter
        else:
            self.maxiter = max_eval//popsize - 1
    
    def solve(self):
        """
        Wrapper function for scipy.optimize.differential_evolution
        """
        progress = []
        def cb(xk, convergence):
            progress.append(self.fun(xk))

        # initialize number of points = popsize
        space = Space([(0.,1.)]*len(self.bounds))
        lhs = Lhs()
        pop = np.asarray(lhs.generate(space.dimensions, self.popsize))
        
        min_b, max_b = np.asarray(self.bounds).T
        diff = max_b - min_b
        pop = min_b + pop * diff
        
        progress.append(np.min(np.apply_along_axis(self.fun, 1, pop)))
        
        result = scipy_de(self.fun, self.bounds, popsize=1, maxiter = self.maxiter, tol=0.0001, disp=self.disp, callback=cb, init=pop)
        self.x = result.x
        self.fx = result.fun
        f_calls = (np.arange(1,len(progress)+1)) * self.popsize
        self.converge_data = np.vstack((f_calls, np.asarray(progress)))
        self.solved = True

    
class SimulatedAnnealing(_Optimizer):
    """
    SimulatedAnnealing class
    """
    def __init__(self, fun, bounds, initial_temp=5230, qv=2.62, popsize=1, get_neighbor='cauchy', temp_func='default', maxiter=5000, max_eval=10000, disp=False):
        # popsize is not relevant for SimulatedAnnealing
        super().__init__(fun, bounds, popsize, maxiter, max_eval, disp)
        self.initial_temp = initial_temp
        self.qv = qv
        if get_neighbor in ('cauchy', 'normal'):
            self.get_neighbor = get_neighbor
        else:
            print("get_neighbor can be either 'cauchy' or 'normal'")
        
        if temp_func =='exponential':
            self.temp_func = SimulatedAnnealing.get_next_temp_exp
        elif temp_func =='ratio':
            self.temp_func = SimulatedAnnealing.get_next_temp_ratio
        elif temp_func =='default':
            self.temp_func = SimulatedAnnealing.get_next_temp_default
        else:
            print("temp_func can be either 'exponential', 'ratio', 'default'") 
            
    @staticmethod
    def get_neighbor_normal(current_state, bounds_range, max_temp, current_temp):
        """
        Randomly selected neighbor of current state drawn from a normal distribution around the current state with a variance dependent on temperature
        
        Parameters
        ----------
        current_state : numpy 1D array
        bounds_range : numpy 1D array
            array([range_x1, range_x2, ...]) where range_xi is a float that is equal to max bound - min bound for feature xi
        max_temp : float
        current_temp : float

        Returns
        -------
        numpy 1D array
        """
        sd_max = bounds_range/np.sqrt(12)
        sd = sd_max/(0.95 * max_temp) * current_temp + 0.05 * sd_max
        return np.random.normal(current_state, sd) 
    
    @staticmethod
    def get_neighbor_cauchy(current_state, bounds_range, min_bounds, max_bounds, 
                  current_temp, qv):
        """
        Randomly selected neighbor of current state drawn from a cauchy distribution based on current state, temperature and qv

        Parameters
        ----------
        current_state : numpy 1D array
        bounds_range : numpy 1D array
            array([range_x1, range_x2, ...]) where range_xi is a float/int that is equal to max bound - min bound for feature xi
        min_bounds : numpy 1D array
            array([min_x1, min_x2, ...]) where min_xi is the min bound for feature xi
        max_bounds : numpy 1D array
            array([max_x1, max_x2, ...]) where max_xi is the max bound for feature xi
        current_temp : float
        qv : float
        Returns
        -------
        x_visit : numpy 1D array
        """
        factor2 = np.exp((4.0 - qv) * np.log(qv - 1.0))
        factor3 = np.exp((2.0 - qv) * np.log(2.0)/(qv - 1.0))
        factor4_p = np.sqrt(np.pi) * factor2 / factor3 * (3.0 - qv)
        factor5 = 1.0 / (qv - 1.0) - 0.5
        d1 = 2.0 - factor5
        factor6 = np.pi * (1.0 - factor5) / np.sin(np.pi * (1.0 - factor5))/ \
        np.exp(gammaln(d1))
        dim = len(current_state)
        
        x, y = np.random.normal(size=(dim, 2)).T
        factor1 = np.exp(np.log(current_temp) / (qv - 1.0))
        factor4 = factor4_p * factor1
        
        x *= np.exp(-(qv - 1.0) * np.log(factor6 / factor4) / (3.0 - qv))
        den = np.exp((qv - 1.0) * np.log(np.fabs(y)) / (3.0 - qv))
    
        visits = x / den
        
        TAIL_LIMIT = 1.e8
        MIN_VISIT_BOUND = 1.e-10
    
        upper_sample, lower_sample = np.random.uniform(size=2)
        visits[visits > TAIL_LIMIT] = TAIL_LIMIT * upper_sample
        visits[visits < TAIL_LIMIT] = TAIL_LIMIT * lower_sample
        x_visit = visits + x
        a = x_visit - min_bounds
        b = np.fmod(a, bounds_range) + bounds_range
        x_visit = np.fmod(b, bounds_range) + min_bounds
        x_visit[np.fabs(x_visit - min_bounds) < MIN_VISIT_BOUND] += 1.e-10
    
        return x_visit
    
    @staticmethod
    def get_next_temp_exp(initial_temp, current_temp, time, qv):
        return initial_temp*np.exp(-time) + 0.000001
    
    @staticmethod
    def get_next_temp_ratio(initial_temp, current_temp, time, qv):
        return current_temp/time**0.5 + 0.000001
    
    @staticmethod
    def get_next_temp_default(initial_temp, current_temp, time, qv):
        return initial_temp*(2**(qv-1)-1)/((1+time)**(qv-1)-1)
    
    def solve(self):
        """
        Solves the optimization problem through simulated annealing algorithm
        """
        
        # provide an initial state and temperature
        time = 0
        current_temp = self.initial_temp

        space = Space([(0.,1.)]*len(self.bounds))
        lhs = Lhs()
        current_state = np.asarray(lhs.generate(space.dimensions, self.popsize))
        min_b, max_b = np.asarray(self.bounds).T
        diff = max_b - min_b
        current_state = min_b + current_state * diff
    
        # evaluate current state
        energy = np.apply_along_axis(self.fun, 1, current_state)
        best_energy = np.min(energy)
        best_state = current_state[np.argmin(energy)]
        evals = self.popsize
        
        # variables for storing progress data
        progress = []
        
        for i in range(self.maxiter):
            for j in range(len(current_state)):
                # generate a new state, randomly chosen neighbour of state
                if self.get_neighbor == 'cauchy':
                    neighbor = SimulatedAnnealing.get_neighbor_cauchy(current_state[j], diff, min_b, max_b, current_temp, self.qv)
                else:
                    neighbor = SimulatedAnnealing.get_neighbor_normal(current_state[j], diff, self.initial_temp, current_temp)
                
                # evaluate new neighbor
                energy_neighbor = self.fun(neighbor)
                delta = energy_neighbor - energy[j]
                evals += 1
           
                if delta < 0:
                    current_state[j] = neighbor
                    energy[j] = energy_neighbor
                    if energy[j] < best_energy:
                        best_energy = energy[j]
                        best_state = current_state[j]
                else:
                    if np.random.rand() < np.exp(-delta/current_temp):
                        current_state[j] = neighbor
                        energy[j] = energy_neighbor
            
            progress.append(best_energy)
            
            time += 1
            current_temp = self.temp_func(self.initial_temp, current_temp, time, self.qv)
            
            if self.disp:
                print(f"simulated annealing step {i}: f(x)= {best_energy}")
                
            if evals > self.max_eval:
                break
        
        f_calls = np.arange(1, i+2) * self.popsize
        self.converge_data = np.vstack((f_calls, np.asarray(progress)))
        self.solved = True
        self.x = best_state
        self.fx = best_energy

      
class ParticleSwarm(_Optimizer):
    def __init__(self, fun, bounds, popsize=20, maxiter=5000, max_eval=10000, disp=False):
        super().__init__(fun, bounds, popsize, maxiter, max_eval, disp)
    
    def solve(self):
        """
        Solves the optimization problem using classic particle swarm algorithm
        """
        # initialize particles with random position and velocity
        
        particles = []
        for i in range(self.popsize):
            particles.append(Particle(self.bounds))
        
        gbest_value = None
        gbest_position = np.empty(len(self.bounds))
        itr = 0
        evals = 0
        
        # variables for storing progress data
        progress = []
        
        while itr < self.maxiter:
            # evaluate cost function and updates personal and global bests
            for i in range(self.popsize):
                particles[i].evaluate(self.fun)
                evals += 1
                if gbest_value == None or particles[i].pbest_value < gbest_value:
                    gbest_value = particles[i].pbest_value
                    gbest_position = particles[i].pbest_position

            # update particles velocity and new positions
            for i in range(self.popsize):
                particles[i].update_velocity(gbest_position)
                particles[i].update_position()
            
            progress.append(gbest_value)
            itr += 1
            if self.disp:
                print(f"simulated annealing step {itr}: f(x)= {gbest_position}")
                
            if evals > self.max_eval - self.popsize:
                break
            
        f_calls = np.arange(1, itr+1)*self.popsize
        self.converge_data = np.vstack((f_calls, np.asarray(progress)))
        self.solved = True
        self.x = gbest_position
        self.fx = gbest_value
        

class Particle:
    """
    Helper class used in ParticleSwarm optimizer
    """
    def __init__(self, bounds, method='latin'):
        self.dimensions = len(bounds)
        self.position = np.empty(self.dimensions)
        self.velocity = np.zeros(self.dimensions)
        self.pbest_position = np.empty(self.dimensions)
        self.pbest_value = None
        self.lowerbounds, self.upperbounds = np.asarray(bounds).T
        
        # initialize positions
        if method == 'random':
            position = np.random.rand(self.dimensions)
        elif method == 'latin':
            space = Space([(0.,1.)]*self.dimensions)
            lhs = Lhs()
            position = np.asarray(lhs.generate(space.dimensions,1))[0]
            
        min_b, max_b = np.asarray(bounds).T
        diff = max_b - min_b
        self.position = min_b + position * diff
    
    def evaluate(self, f):
        cost = f(self.position)

        if self.pbest_value == None:
            self.pbest_value = cost
        elif cost < self.pbest_value:

            if sum(self.upperbounds < self.position)==0 and sum(self.lowerbounds > self.position)==0: 
                self.pbest_value = cost
                self.pbest_position = self.position
    
    def update_velocity(self, gbest, inertia=0.5, c1=1, c2=2):
        # c1: cognitive parameter, c2: social parameter
        self.velocity = inertia*self.velocity + \
                        c1*np.random.rand()*(self.pbest_position-self.position) + \
                        c2*np.random.rand()*(gbest-self.position)
    
    def update_position(self):
        self.position = self.position + self.velocity
        
        