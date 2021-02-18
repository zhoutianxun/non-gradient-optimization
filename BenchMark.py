import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_2d(function, bounds=[(-10,10),(-10,10)], style='3D'):
    """
    Parameters
    ----------
    function : function accepts 1D numpy array of arbitrary length and outputs single value
    bounds : length 2 list of 2D tuple, optional
        The default is [(-10,10),(-10,10)].
    style: string, optional
        The default is 3D. Other option is 'contour'
    """
    try:
        x = np.linspace(bounds[0][0], bounds[0][1])
        y = np.linspace(bounds[1][0], bounds[1][1])
        xv, yv = np.meshgrid(x, y)
    except (IndexError, NameError):
        print("Bounds are not correct. Needs to be length 2 list of 2D tuples i.e. [(x_lb, x_ub), (y_lb, y_ub)]")
        return
    
    try:
        z = np.apply_along_axis(function, 0, np.vstack((np.ravel(xv), np.ravel(yv))))
        z = z.reshape(xv.shape)
    except:
        print("Please check function")
        return
        
    fig = plt.figure()
    if style == '3D':
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection="3d")
        surf = ax.plot_surface(xv, yv, z, cmap=cm.coolwarm)
        fig.colorbar(surf)
    elif style == 'contour':
        ax = plt.axes()
        ax.contourf(x, y, z, cmap=cm.coolwarm)
    else:
        print('Invalid style')
    plt.show()

def ackley(x):
    """
    f(x*) = 0 at x* = (0,...,0)

    Parameters
    ----------
    x : numpy array

    """
    return -20*np.exp(-0.2*np.sqrt(1/len(x)*sum(x**2))) - np.exp(1/len(x)*sum(np.cos(2*np.pi*x))) + 20 + np.exp(1)

def griewank(x):
    """
    f(x*) = 0 at x* = (0,...,0)

    Parameters
    ----------
    x : numpy array

    """
    np.arange(1,len(x)+1)
    b = np.cos(x/np.sqrt(np.arange(1,len(x)+1)))
    return sum(x**2/4000) + np.dot(b, b.T) + 1

def levy(x):
    """
    f(x*) = 0 at x* = (1,...,1)

    Parameters
    ----------
    x : numpy array

    """
    w = 1 + (x-1)/4
    a = np.sin(np.pi*w[0])**2
    b = (w-1)**2*(1+10*np.sin(np.pi*w+1)**2)
    c = (w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2)
    return a + sum(b+c)
    
def michale(x):
    """
    f(x*) = ? at x* = ?

    Parameters
    ----------
    x : numpy array

    """
    return -sum(np.sin(x)*np.sin(np.arange(1,len(x)+1)*x**2/np.pi)**(20))

def rastrigin(x):
    return sum(10+x**2-10*np.cos(2*np.pi*x))

def rosen(x):
    """
    f(x*) = 0 at x* = (1,...,1)
    
    Parameters
    ----------
    x : numpy array

    """
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def sphere(x):
    """
    f(x*) = 0 at x* = (0,...,0)

    Parameters
    ----------
    x : numpy array

    """
    return sum(x**2)

def zakharov(x):
    b = sum(0.5*np.arange(1,len(x)+1)*x)
    return sum(x**2) + b**2 + b**4
