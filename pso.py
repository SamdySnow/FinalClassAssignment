import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from tqdm import tqdm 
def init(item_weight,item_value,b_=700,xSize_=200,iteration_=1000,c1_=0.5,c2_=0.5,w_=0.8):
    global a,c,b,Dim,xSize,iteration,c1,c2,w,A,C,x,v,xbest,fxbest,gbest,fgbest
    a = item_weight #物品重量
    c = item_value  #物品价值
    #a = [90, 33, 94, 69, 77, 91, 39, 74, 24, 34, 14, 89, 98, 37, 32, 45, 15, 98, 40, 16, 17, 4, 3, 5, 94, 3, 64, 47, 85, 9, 6, 39, 44, 67, 33, 59, 17, 16, 55, 95, 69, 88, 91, 28, 66, 54, 85, 82, 24, 17, 30, 66, 96, 8, 74, 20, 84, 35, 53, 19, 25, 64, 98, 93, 86, 24, 30, 68, 56, 37, 6, 98, 48, 76, 61, 9, 29, 76, 55, 41, 93, 19, 56, 85, 20, 84, 12, 64, 94, 29, 26, 93, 83, 72, 76, 86, 4, 99, 29, 4]
    #c = [68, 20, 125, 85, 113, 109, 19, 93, 17, 19, 18, 79, 92, 39, 45, 50, 12, 77, 53, 16, 8, 2, 2, 7, 47, 4, 67, 9, 62, 6, 8, 51, 61, 93, 33, 35, 14, 15, 34, 136, 87, 72, 121, 28, 84, 54, 110, 44, 15, 23, 36, 95, 77, 11, 74, 14, 75, 50, 63, 26, 34, 56, 67, 77, 73, 34, 41, 59, 84, 44, 7, 112, 70, 65, 54, 6, 20, 92, 56, 21, 125, 24, 83, 113, 14, 99, 18, 49, 70, 15, 34, 88, 105, 48, 61, 128, 3, 110, 19, 4] 
    b = b_  #背包容量
    #初始化种群
    Dim = len(a)           #维度
    xSize = xSize_         #种群数
    iteration = iteration_        #迭代次数
    c1 = c1_
    c2 = c2_           #加速因子
    w = w_            #定义惯性因子 
    A = np.array([a]*xSize)                #将a扩展成种群数*维度的矩阵  
    C = np.array([c]*xSize)                #将c扩展为种群数*维度的矩阵
    x = np.random.randint(0,2,(xSize,Dim)) #随机生成一个种群数*维度的矩阵
    v = np.random.rand(xSize,Dim)          #随机生成一个种群数*维度的速度矩阵
    xbest = np.zeros((xSize,Dim))            #单个粒子的初始最佳位置
    fxbest = np.zeros((xSize,1))             #xbext的适应度
    gbest = np.zeros((1,Dim))                #全局最优解
    fgbest = 0                             #全局最优解的适应度        

def solve():   
    #寻找粒子群最优位置和单个粒子
    global x,fgbest,v
    fx = np.sum(np.multiply(C,x), axis=1) # 粒子适应度，即背包内物品的价格
    sx = np.sum(np.multiply(A,x), axis=1) # 限制函数，即背包内物品的体积
    #print(sx)
    for i in range(xSize):
        if list(sx)[i] > b:
            fx[i] = 0
    for i in range(xSize):
        if fxbest[i] < fx[i]:   # 当粒子适应度大于最佳适应度时，替代
            fxbest[i] = fx[i]
            xbest[i] = x[i] # 替换矩阵第i行
    if fgbest <= max(fxbest):
        fgbest = max(fxbest)
        g = list(fxbest).index(fgbest)
        gbest = xbest[g]     #当存在粒子的最佳适应度fxbext(i)大于种群最佳适应度fgbext(i)时，替代
    for i in range(xSize):
        if (x[i]==gbest).all():
            x[i] = np.random.randint(0,2,(1,Dim)) #随机生成一个种群数*维度的矩阵
    R1 = np.random.rand(xSize,Dim)
    R2 = np.random.rand(xSize,Dim)
    v = v * w  + c1 * np.multiply(R1,xbest-x) + c2 * np.multiply(R2,(np.array([gbest]*xSize)-x))#速度迭代公式产生新的速度
    x = x + v
    for i in range(xSize):   #更新粒子群的位置
        for j in range(Dim):
            if x[i][j] < 0.5:
                x[i][j] = 0
            else:
                x[i][j] = 1   #粒子的位置只有（0,1）两种状态
    a = copy.deepcopy(fgbest)
    return a

def pso(item_weight,item_value,b_=700,xSize_=200,iteration_=1000,c1_=0.5,c2_=0.5,w_=0.8):
    """
    PSO algorythm to solve 0-1kp

    Parameters
    ----------
    item_weight:list or np.ndarray
        a list of items' weight 
    item_value:list or np.ndarray
        a list of items' value 
    b_:int
        the size of the package
    individuals:int
        initial number of individuals, if not get value, it will be round(weight/2 - 0.5)
    xSize_:int
        how many populations do you want
    iteration_:int
        times of iterate
    c1_ and c2_:int
        two boost consts
    w_:int
        inertance const

    Returns
    -------
    """
    init(item_weight,item_value,b_,xSize_,iteration_,c1_,c2_,w_)
    fgbest_list = []
    for i in tqdm(range(iteration_)):
        a = solve()
        fgbest_list.append(a)
    plt.plot(fgbest_list)
    plt.show()

