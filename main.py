import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm 
import copy

def bag(n,c,w,v):
	res=[[-1 for j in range(c+1)] for i in range(n+1)]
	for j in range(c+1):
		res[0][j]=0
	for i in tqdm(range(1,n+1)):
		for j in range(1,c+1):
			res[i][j]=res[i-1][j]
			if j>=w[i-1] and res[i][j]<res[i-1][j-w[i-1]]+v[i-1]:
				res[i][j]=res[i-1][j-w[i-1]]+v[i-1]
	return res
 
def show(n,c,w,res):
	print('最大价值为:',res[n][c])
	x=[False for i in range(n)]
	j=c
	for i in range(1,n+1):
		if res[i][j]>res[i-1][j]:
			x[i-1]=True
			j-=w[i-1]
	print('选择的物品为:')
	for i in range(n):
		if x[i]:
			print('第',i,'个,',end='')
	print('')
 
if __name__=='__main__':
	n=1000 #样本总量
	c=5000 #背包大小
	w=np.random.randint(1,200,1000) #宝物重量
	v=np.random.randint(1,500,1000) #宝物价值
	res=bag(n,c,w,v)
	show(n,c,w,res)
    