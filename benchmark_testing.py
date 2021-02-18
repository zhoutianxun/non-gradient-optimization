from BenchMark import *
from Optimizers2 import *

# Define function test bounds
fun_bounds = dict(ackley_bounds = [(-32.7, 32.7)],
                  griewank_bounds = [(-600, 600)],
                  levy_bounds = [(-10, 10)],
                  michale_bounds = [(0, 3.141592654)],
                  rastrigin_bounds = [(-5.12, 5.12)],
                  rosen_bounds = [(-10, 10)],
                  sphere_bounds = [(-5.12, 5.12)],
                  zakharov_bounds = [(-5, 10)])

functions = [ackley,
             griewank,
             levy,
             michale,
             rastrigin,
             rosen,
             sphere,
             zakharov]

def plotting(title, **kwargs):
    """
    Plots all inputs on the same plot, see parameter:
    
    Parameters
    ----------
    *args : list
        list of arbitrary length, each element is a 2 x n numpy array
        the 1st row of the numpy array is the x values
        the 2nd row of the numpy array is the y values
    """
    for key, value in kwargs.items():
       plt.plot(value[0], value[1], label=key)
    
    plt.xlabel('function evaluations')
    plt.ylabel('function cost')
    plt.legend()
    plt.title(title)
    plt.show()

popsize = 20

for fun in functions:
    for test_cond in [(10, 1000), (30, 1000), (50, 1000)]:
        d = test_cond[0]
        max_eval = test_cond[1]
        bounds = fun_bounds[fun.__name__ + '_bounds'] * d
        
        # test out 3 optimizers 
        DE = DifferentialEvolution(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
        DE.solve()
        DE_plot = DE.converge_data
        
        SA = SimulatedAnnealing(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
        SA.solve()
        SA_plot = SA.converge_data
        
        PS = ParticleSwarm(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
        PS.solve()
        PS_plot = PS.converge_data
        
        title = f"{fun.__name__} with {d} dimensions and {max_eval} function evaluations"
        plotting(title, DE=DE_plot, SA=SA_plot, PS=PS_plot)
        
        input("Press Enter to continue...")
        
        plt.close()
        
    
        


