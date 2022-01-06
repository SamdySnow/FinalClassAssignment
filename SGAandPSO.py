"""
SGAandPSO
---------
provide SGA() and DPSO()

do not use other functions as far as possible
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def initial(individuals,bits):
    return np.random.randint(0,2,(individuals,bits),dtype=int)

def fitness(individuals,weight,price,package_weight):
    price_res = np.zeros(len(individuals))
    for i,individual in enumerate(individuals):
        temp_weight = 0
        temp_price = 0
        for index in np.nonzero(individual)[0]:
            temp_weight = temp_weight + weight[index]
            temp_price = temp_price + price[index]
        price_res[i] = temp_price if temp_weight <= package_weight else 1e-15
    return price_res

def roulette_wheel_selection(individuals,fit_res):
    index = np.random.choice(np.arange(len(individuals)), size=len(individuals), replace=True,p=(fit_res)/(fit_res.sum()))
    return individuals[index]

def two_point_crossover(individuals,bits,group,crossover_rate = 0.8):
    new_group = np.zeros((individuals,bits),dtype=int)
    for ind,individual1 in enumerate(group):
        new_individual = individual1
        if np.random.rand() <= crossover_rate:
            individual2 = group[np.random.randint(individuals)]
            m,n = np.random.choice(bits,2,False)#选择起始点和结束点
            if m > n:
                n,m = m,n
            new_individual[m:n] = individual2[m:n]
        new_group[ind] = new_individual
    return new_group

def point_variation(bits,group,p=0.005):
    for individual in group:
        if np.random.rand() <= p:
            point_index = np.random.randint(bits)
            individual[point_index] = 1 - individual[point_index]
    return group   

def update_vx(v,x,p_best,g_best):
    new_x = x.copy()
    for k,ind in enumerate(v):
        for i in range(len(ind)):
            ind[i] = ind[i] + np.random.rand()*(p_best[k][i] - ind[i]) + np.random.rand()*(g_best[i] - ind[i])
            if np.random.rand() <= 1/(1+math.exp(-ind[i])):
                new_x[k][i] = 1
            else:
                new_x[k][i] = 0
    return v,new_x

def update_pg(x,fit_res,p_best,g_best,w,p,pa):
    temp_fit = fitness(p_best,w,p,pa)
    best_fit = fitness([g_best],w,p,pa)[0]
    for i in range(len(p_best)):
        if fit_res[i] > temp_fit[i]:
            p_best[i] = x[i]
    temp_arg_g = np.argmax(temp_fit)
    if np.max(temp_fit) > best_fit:
        g_best = x[temp_arg_g]
    return p_best,g_best


def SGA(package_weight,weight,price,individuals = None,iters = 100,crossover=0.8,variation=0.005):
    """
    SGA(package_weight,weight,price,individuals = None,iters = 100,crossover=0.8,variation=0.005)

    警告：没有测试，有问题自己看着改，改不了再说warning:not tested

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
    crossover:float
        the rate of ceossover
    variation:float
        the rate of variation

    Returns
    -------
    the best resolve:float
        depand on input
    """
    dimension = len(weight)
    init_individual = individuals if individuals != None else round(dimension/2-0.5) # 初始化种群数量
    try:
        weight = np.array(weight)
        price = np.array(price)
    except:
        raise "weight or price not array-like"
    end_max = []
    end_ave = []
    init_group = initial(init_individual,dimension)
    fit_result = fitness(init_group,weight,price,package_weight)
    end_max.append(np.max(fit_result))
    end_ave.append(np.mean(fit_result))
    for it in range(iters):
        next_group = roulette_wheel_selection(init_group,fit_result)
        new_group = point_variation(dimension,two_point_crossover(init_individual,dimension,next_group,crossover),variation)
        fit_result = fitness(new_group,weight,price,package_weight)
        end_max.append(np.max(fit_result))
        end_ave.append(np.mean(fit_result))
        init_group = new_group
    plt.plot(end_max,'r-')
    plt.plot(end_ave,'b-')
    plt.xlabel('iteration')  
    plt.ylabel('fitness')  
    plt.title('fitness curve')  
    plt.show()
    return np.array(np.max(end_max))

def DPSO(package_weight,weight,price,individuals = None,iters = 100):
    """
    DPSO(package_weight,weight,price,individuals = None,iters = 100)
    
    警告：没有测试，有问题自己看着改，改不了再说warning:not tested

    DPSO algorythm to solve 0-1kp

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
    crossover:float
        the rate of ceossover
    variation:float
        the rate of variation

    Returns
    -------
    the best resolve:float
        depand on input
    """
    dimension = len(weight)
    init_individual = individuals if individuals != None else round(dimension/2-0.5) # 初始化种群数量
    try:
        weight = np.array(weight)
        price = np.array(price)
    except:
        raise "weight or price not array-like"
    end_max = []
    end_ave = []
    init_group = initial(init_individual,dimension)
    fit_result = fitness(init_group,weight,price,package_weight)
    end_max.append(np.max(fit_result))
    end_ave.append(np.mean(fit_result))
    p_best = np.zeros((individuals,dimension))
    g_best = np.zeros(dimension)
    p_best = init_group
    g_best = init_group[np.argmax(fit_result)]
    speed = np.random.randint(0,10,(individuals,dimension))/10
    for it in range(iters):
        new_speed,new_x = update_vx(speed,init_group,p_best,g_best)
        new_fit = fitness(new_x,weight,price,package_weight)
        for i in range(len(new_fit)):
            if new_fit[i] > fit_result[i]:
                init_group[i] = new_x[i]
        p_best,g_best = update_pg(new_x,new_fit,p_best,g_best,weight,price,package_weight)
        end_max.append(np.max(new_fit))
        end_ave.append(np.mean(new_fit))
        speed = new_speed
        fit_result = new_fit
    plt.plot(end_max,'r-')
    plt.plot(end_ave,'b-')
    plt.xlabel('iteration')  
    plt.ylabel('fitness')  
    plt.title('fitness curve')  
    plt.show()
    return np.array(np.max(end_max))
