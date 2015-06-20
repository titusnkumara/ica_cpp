import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import random
from sklearn.decomposition import FastICA, PCA

print "Program started"

dimnsion = int(sys.argv[1])
n_samples = int(sys.argv[2])
blocking  = str(sys.argv[3])

###############################################################################
# Generate sample data
alpha, beta = 10,5 # alpha beta values
np.random.seed(0)

ti = np.linspace(0, 20, n_samples)

s1 = np.sin(2 * ti)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * ti))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * ti)  # Signal 3: saw tooth signal

genList = []
for k in range(0,dimnsion-3):
    rnd = random.random()*random.randint(1,15)
    s5 = np.sin(rnd*ti)
    for i in range(0,random.randint(5,50)):
        tmpRandom = random.random()*random.randint(1,15)
        s5 = s5+np.sin(tmpRandom * ti)
    print 'iterated ', i
    s5 = s5*0.1
    genList.append(s5)
ar = [s1, s2, s3]
for i in genList:
    ar.append(i)

S = np.array(ar).T
#S = np.c_[s1, s2, s3,s4,genList[0]] #,genList[1],genList[2],genList[3],genList[4],genList[5]]

std = S.std(axis=0)  # Standardize data

# Mix data
#A = np.array([[1, 1, 1,1,1], [0.5, 2, 1.0,0.7,2.1], [1.5, 1.0, 2.0,1.3,1.2],[1.2,0.3,0.7,2.5,3.1],[0.1,0.5,0.1,4.3,1.7]])  # Mixing matrix
A = np.random.rand(dimnsion,dimnsion)
X = np.dot(S, A.T)  # Generate observations

#write data
fp = open("data.txt","w")
Data =  X.T
for row in Data:
    for item in row:
        fp.write(str(item)+'\t')
    fp.write('\n')
fp.close()


plt.figure()

models = [X, S]
names = ['Observations (mixed signal)',
         'True Sources']
colors = ['red', 'steelblue', 'orange','black','green','gray','gold','coral','plum','crimson']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show(block=(blocking=='block'))
