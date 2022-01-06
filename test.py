# just used to test in debug etc. do not edit or run.
import SGAandPSO as sp
import matplotlib.pyplot as plt
import numpy as np
import pso
import json
f = json.load(open("testdata/testdata(9).json"))
data = f["data8"]
weight = data["weight"]
price = data["price"]

pso.pso(weight,price,b_=1173,iteration_=300)
