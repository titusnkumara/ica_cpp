import sys
import numpy as np


X1 = np.loadtxt("pythonout.txt")
X2 = np.loadtxt("result.txt")

np.set_printoptions(formatter={'float': '{: 0.10f}'.format})

i = np.array([0 ,2, 1])
X2 =X2[:, i]

print X2

print X1-X2
