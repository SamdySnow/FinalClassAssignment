import numpy as np
import matplotlib.pyplot as plt

class SGA(object):

    def __init__(self,init_bit,init_individual,iter_number):
        self.init_bit = init_bit  # 编码长度
        self.init_individual = init_individual  # 初始化种群数量
        self.iteration = iter_number  # 迭代次数
        self.bag_capacity = 200  # 背包容量
        self.remain = 0.2 #保留精英
        self.remain_rate = 0.5 #其余保留概率？
        self.variation_rate = 0.005 #变异概率

    def initial(self):
        return np.random.randint(0,2,(self.init_individual,self.init_bit),dtype=np.int32)

    def fitness(self,individual):
        weight = 0
        price = 0
        