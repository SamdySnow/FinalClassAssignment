# just used to test in debug etc. do not edit or run.
import SGAandPSO as sp
import numpy as np
import json
data = json.load(open(r'testdata/testdata(9).json'))
print(sp.DPSO(data['data4']['package_weight'],data['data4']['weight'],data['data4']['price'],10,50))