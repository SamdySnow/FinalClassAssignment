import SGA as sp
import SA as sa
import pso as pso
import greed as gr
import FA as fa
import numpy as np
import json
import time
from matplotlib import pyplot as plt

if __name__=='__main__':
	data = json.load(open(r'testdata/testdata(9).json'))
	data_set = []
	for i in range(1,10):
		data_set.append(data["data"+str(i)])
	times = []
	values = []
	current_data = {"package_weight":8000,"weight":np.random.randint(1,200,300),"price":np.random.randint(1,200,300)}
	for i in range(4):
		best = 0
		starttime = time.time()
		if i == 0:
			value = gr.bag(current_data["package_weight"],current_data["weight"],current_data["price"])
		elif i == 1:
			value = pso.pso(current_data["weight"],current_data["price"],current_data["package_weight"],20,200)
		elif i == 2:
			value = sa.SAA(current_data["weight"],current_data["price"],current_data["package_weight"],100,time=50)
		elif i == 3:
			value = fa.pso(current_data["weight"],current_data["price"],current_data["package_weight"],20,200)
		endtime = time.time()
		times.append(endtime-starttime)
		values.append(value)
	plt.subplot(1,2,1)
	plt.ylabel("best value")
	plt.bar(["greedy","PSO","SA","FA"],values)
	plt.subplot(1,2,2)
	plt.ylabel("time")
	plt.bar(["greedy","PSO","SA","FA"],times)
	plt.show()