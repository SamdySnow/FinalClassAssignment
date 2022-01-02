import numpy as np
import matplotlib.pyplot as plt

def initial(individuals,bits):
    return np.random.randint(0,2,(individuals,bits),dtype=int)

def fitness(individuals,weight,price,package_weight):
    price_res = np.zeros(len(weight))
    for i,individual in enumerate(individuals):
        temp_weight = 0
        temp_price = 0
        for index in np.nonzero(individual):
            temp_weight += weight[index]
            temp_price += price[index]
        price_res[i] = temp_price if temp_weight <= package_weight else 0
    return price_res

def roulette_wheel_selection(individuals,fit_res):
    index = np.random.choice(np.arange(individuals), size=individuals, replace=True,p=(fit_res)/(fit_res.sum()))
    return individuals[index]


        

def SGA(package_weight,weight,price,individuals = None,iters = 100,best = None):
    """
    SGA algorythm to solve 0-1kp

    Parameters
    ----------
    package_weight:int
        total weight of package  
    weight:list or np.ndarray
        weight of jewel  
    price:list or np.ndarray
        price of jewel, which should correspond to weight  
    individuals:int
        initial number of individuals, if not get value, it will be round(weight/2 - 0.5)
    iter:int
        iter times, default 100
    best:int,optional
        if know the best value, then test the iters that find the value

    Returns
    -------
    """
    dimension = len(weight)
    init_individual = individuals if individuals != None else round(dimension/2-0.5) # 初始化种群数量
    try:
        weight = np.array(weight)
        price = np.array(price)
    except:
        raise "weight or price not array-like"
    # remain = 0.2 #保留精英
    # remain_rate = 0.5 #其余保留概率？
    # variation_rate = 0.005 #变异概率
    init_group = initial(init_individual,dimension)
    fit_result = fitness(init_group,weight,price,package_weight)
    for it in range(iters):
        pass