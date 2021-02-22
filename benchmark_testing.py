from BenchMark import *
from Optimizers import *

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
        the 2nd to nth row of the numpy array are the y values of different tries
    """
    # plotting colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    i=0
    
    fig, axes = plt.subplots()
    for key, value in kwargs.items():
        x = value[0]
        y_mean = np.apply_along_axis(np.mean, 0, value[1:])
        axes.plot(x, y_mean, label=key)
        print("Computing quantiles...")
        box_plot(value[1:,::5], positions=x[::5]+i*5, axes=axes, widths=10, edge_color=colors[i], showfliers=False)
        i += 1
        
        print(f"{key} function costs: {np.vstack((x,y_mean))}")
    axes.set_xlabel('function evaluations')
    axes.set_ylabel('function cost')
    axes.legend()
    axes.set_title(title)
    plt.show()

def box_plot(value, positions, axes, widths, edge_color, fill_color='None', showfliers=False):
    bp = axes.boxplot(value, positions=positions, widths=widths, showfliers=showfliers, manage_ticks=False, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 


def max_eval_testing(test_conds=[(10, 500), (30, 500), (50, 500)], popsize=20, runs=20):
    for fun in functions:
        for test_cond in test_conds:
            d = test_cond[0]
            max_eval = test_cond[1]
            bounds = fun_bounds[fun.__name__ + '_bounds'] * d
            
            print(f"Run number: 1")
            DE = DifferentialEvolution(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
            DE.solve()
            DE_plot = DE.converge_data
             
            SA = SimulatedAnnealing(fun, bounds, popsize=1, maxiter=max_eval, max_eval=max_eval)
            SA.solve()
            SA_plot = SA.converge_data[:,popsize-1::popsize]
            
            PS = ParticleSwarm(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
            PS.solve()
            PS_plot = PS.converge_data        
        
            for run in range(runs-1):
                print(f"Run number: {run+2}")
                # test out 3 optimizers 
                DE = DifferentialEvolution(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
                DE.solve()
                DE_plot = np.vstack((DE_plot, DE.converge_data[1]))
                
                SA = SimulatedAnnealing(fun, bounds, popsize=1, maxiter=max_eval, max_eval=max_eval)
                SA.solve()
                SA_plot = np.vstack((SA_plot, SA.converge_data[1,popsize-1::popsize]))
                
                PS = ParticleSwarm(fun, bounds, popsize=popsize, maxiter=max_eval, max_eval=max_eval)
                PS.solve()
                PS_plot = np.vstack((PS_plot, PS.converge_data[1]))
            

            title = f"{fun.__name__} with {d} dimensions and {max_eval} function evaluations"
            plotting(title, DE=DE_plot, SA=SA_plot, PS=PS_plot)
            
            input("Press Enter to continue...")
            
            plt.close()
        
def max_iter_testing(test_conds=[(10, 10), (30, 10), (50, 10)], popsize=100, runs=20):
    for fun in functions:
        for test_cond in test_conds:
            d = test_cond[0]
            max_iter = test_cond[1]
            bounds = fun_bounds[fun.__name__ + '_bounds'] * d
            
            print(f"Run number: 1")
            DE = DifferentialEvolution(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
            DE.solve()
            DE_plot = DE.converge_data
             
            SA = SimulatedAnnealing(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
            SA.solve()
            SA_plot = SA.converge_data
            
            PS = ParticleSwarm(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
            PS.solve()
            PS_plot = PS.converge_data        
        
            for run in range(runs-1):
                print(f"Run number: {run+2}")
                # test out 3 optimizers 
                DE = DifferentialEvolution(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
                DE.solve()
                DE_plot = np.vstack((DE_plot, DE.converge_data[1]))
                
                SA = SimulatedAnnealing(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
                SA.solve()
                SA_plot = np.vstack((SA_plot, SA.converge_data[1]))
                
                PS = ParticleSwarm(fun, bounds, popsize=popsize, maxiter=max_iter, max_eval=1e9)
                PS.solve()
                PS_plot = np.vstack((PS_plot, PS.converge_data[1]))

            title = f"{fun.__name__} with {d} dimensions and {max_iter} function evaluations"
            plotting(title, DE=DE_plot, SA=SA_plot, PS=PS_plot)
            
            input("Press Enter to continue...")
            
            plt.close()


