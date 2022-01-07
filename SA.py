import random
import math
import copy
import numpy as np

def calc_price(solv,price):
    #计算背包价值
    res = 0.0
    for i in range(len(solv)):
        if solv[i]:
            res += price[i]
    
    return res

def calc_weight(slove,weight):
    #计算背包重量
    res = 0.0
    for i in range(len(slove)):
        if slove[i]:
            res += weight[i]
    
    return res


def SAA(weight = None, price = None, pw = 0, T = 200, af = 0.95, time = 20, balance = 5, bestt = None):
    '''
    模拟退火算法
    ------------------------------------------------
    输入：
    ------------------------------------------------
    重量weight[dimensionality]
    
    价值price[dimensionality]

    背包容量pw，默认为 0

    初始温度T，默认为 200

    退火率af 默认为 0.95

    迭代次数time 默认为 20

    平衡次数balance 默认为 5
    
    ------------------------------------------------
    输出：
    ------------------------------------------------
    解集res[dimensionality]
    '''
    init = [0] * len(weight)
    c = len(init) - 1
    best = copy.deepcopy(init) #全局最优解
    now = copy.deepcopy(init)

    for i in range(time):

        now_price = calc_price(now,price) #当前背包重量
        #print('i = ',i)
        

        for j in range(balance):
            #print('j = ',j)
            test = copy.deepcopy(now)

            r = random.randint(0,c) #随机位置

            if test[r] == 0:
                #若物品不在背包中
                test[r] = 1  # 直接放入背包
                if random.random() < 0.5:
                    
                    #或代替一件物品
                    while(1):
                        ob = random.randint(0, c)
                        if(test[ob] == 1):
                            test[ob] = 0
                            break
            else:
                #若已经在背包中
                test[r] = 0 #将其取出
                while(1): #并放入另一件物品
                    ob = random.randint(0, c)
                    if(test[ob] == 0):
                        test[ob] = 1
                        break
            #计算合法性
            if calc_weight(test,weight) > pw: #超重
                continue #直接丢弃
            else:
                if calc_price(test,price) > now_price: #优于当前解
                    now = copy.deepcopy(test) #直接接受新解
                    if calc_price(best,price) < calc_price(test,price):
                        best = copy.deepcopy(test) #更新全局最优解
                else: #劣于当前解
                    g = 1.0*(calc_price(test,price) - now_price)/T
                    if(random.random() < math.exp(g)):  # 概率接受劣解
                        now = copy.deepcopy(test)
            
        T = T*af #温度下降
        if T < 10:
            #print('到达最小温度阈值')
            break #到达最小温度阈值
        
        if bestt:
            if np.abs(now_price-bestt) <= 0.01*bestt:
                break
    return calc_price(best,price)
            
                        
if __name__ == '__main__':
    #使用示例

    weight = [135, 133, 130, 11, 128, 123, 20, 75, 9, 66, 105, 43, 18, 5, 37, 90, 22, 85, 9, 80, 70, 17, 60, 35, 57, 35, 61, 40,
              8, 50, 32, 40, 72, 35, 100, 2, 7, 19, 28, 10, 22, 27, 30, 88, 91, 47, 68, 108, 10, 12, 43, 11, 20, 37, 17, 4, 3, 21, 10, 67]
    price = [199,194,193,191,189,178,174,169,164,164,161,158,157,154,152,152,149,142,131,125,124,124,124,122,119,116,114,113,111,110,109,100,97,94,91,82,82,81,80,80,80,79,77,76,74,72,71,70,69,68,65,65,61,56,55,54,53,47,47,46,41,36,34,32,32,30,29,29,26,25,23,22,20,11,10,9,5,4,3,1]
    init = [0] * len(weight)
    pw = 1173

    best = SAA(init=init, weight=weight, price=price, pw=pw, time=1000, T=50000,af = 0.95,balance=20)
    print('最优解为',calc_price(best,price))
    print('背包重量',calc_weight(best,weight))
    SAA()
        
            


 
    
