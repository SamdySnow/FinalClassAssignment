import numpy as np
import matplotlib.pyplot as plt
import sys

def initial(self):
        return np.random.randint(0,2,(self.init_individual,self.init_bit),dtype=np.int32)

def fitness(self,individual):
    weight = 0
    price = 0


if __name__ == '__main__':

    # init_bit =   # 编码长度
    # init_individual =   # 初始化种群数量
    # iteration =  # 迭代次数
    # print(init_bit,init_individual,iteration)
    bag_capacity = 200  # 背包容量
    remain = 0.2 #保留精英
    remain_rate = 0.5 #其余保留概率？
    variation_rate = 0.005 #变异概率

    
        