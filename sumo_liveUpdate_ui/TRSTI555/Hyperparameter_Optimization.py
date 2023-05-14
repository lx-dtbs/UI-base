
from bayes_opt import BayesianOptimization
from losser5 import run_R_TuRBO

global flag
flag = 0

def TRSTI_Algorithm(x):
    print("外部超参数下一迭代点为")
    print(x)
    global flag
    flag = flag + 1

    ### 调用TRSTI算法优化
    MinWT, x_best = run_R_TuRBO(x,flag)
    
    print("第{0}次迭代目标函数值为：{1}".format(flag,MinWT))
    print("其所对应的信号配时方案为：{}\n".format(x_best))
    
    return -MinWT
        
# Bounded region of parameter space
pbounds={"x": (0.1, 10)}

optimizer = BayesianOptimization(
    f=TRSTI_Algorithm,
    pbounds=pbounds,
    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=3,
    n_iter=30,
    acq='ucb',
    xi=10
)

### While the list of all parameters probed and their corresponding target values is available via the property bo.res.
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

### The best combination of parameters and target value found can be accessed via the property bo.max.
print("最好的结果为")
print(optimizer.max)


