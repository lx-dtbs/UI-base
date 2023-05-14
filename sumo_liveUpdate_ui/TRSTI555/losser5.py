
from turbo import Turbo1
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import sumo
import Transform_sigal_timing
import pandas as pd
import timeit
sumo_iter = 0
SUMO_time = []
iter_x = []
iter_y = []

# Set up an optimization problem class
class SUMO_Function:
    def __init__(self, dim=10, u=1):
        self.dim = dim
        self.lb = 5 * np.ones(dim)
        self.ub = 35 * np.ones(dim)
        self.u = u
        
    def __call__(self, x):
        global iter_x
        iter_x.append(x)
        print(x)
        start1 = timeit.default_timer()
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        Transform_sigal_timing.Green_duration(x)
        y, y_SP = sumo.myfunction(x, self.u)
        ### self.TS_weight = np.append(self.TS_weight, y_SP.reshape(-1,dim), axis=0)
        global iter_y
        iter_y.append(float(y))
        stop1 = timeit.default_timer()
        each_SUMO_time = stop1 - start1
        global SUMO_time
        SUMO_time.append(each_SUMO_time)
        global sumo_iter
        sumo_iter += 1
        print('sumo成功运行', sumo_iter)
        return float(y), y_SP


def run_R_TuRBO(u,flag):
    global sumo_iter
    sumo_iter = 0
    f = SUMO_Function(27,u)

    # Create a Turbo optimizer instance
    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=5,  # Number of initial bounds from an Latin hypercube design
        max_evals = 10,  # Maximum number of evaluations
        batch_size=5,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    ###计时开始
    start = timeit.default_timer()

    # Run the optimization process
    turbo1.optimize()

    # Extract all evaluations from Turbo and print the best
    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

    ###计时结束
    stop = timeit.default_timer()
    T = stop - start
    print('网格形网络总迭代时间为'+str(T))

    # 存储每次迭代数据
    data = pd.DataFrame({'X':iter_x,'Y':iter_y,'sumo_time':SUMO_time})
    data.to_csv('Turbo_{0}_{1}_{2}_{3}.csv'.format(27,10,50,flag),index = None,encoding = 'utf8')

    # Plot the progress
    fig = plt.figure(figsize=(7, 5))
    matplotlib.rcParams.update({'font.size': 16})
    plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
    plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
    plt.xlim([0, len(fX)])
    plt.ylim([min(iter_y)-min(iter_y)/10, max(iter_y)+max(iter_y)/10])
    plt.title("27D SUMO function")

    plt.tight_layout()
    plt.savefig('{}result.jpg'.format(flag))
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    
    return f_best[0], x_best

