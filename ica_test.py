import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

from sklearn.decomposition import FastICA, PCA

###############################################################################
# Generate sample data
alpha, beta = 10,5 # alpha beta values
np.random.seed(0)
n_samples = sys.argv[1]
ti = np.linspace(0, 10, n_samples)

s1 = np.sin(2 * ti)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * ti))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * ti)  # Signal 3: saw tooth signal

#print "initial data arrays"
#print s1
#print s2
#print s3

S = np.c_[s1, s2, s3]




'''
Here we should combine the arrays into one matrix
it is 10*3 matrix

'''

#print '\ncombined arrays'
#print S,S.shape

'''
Then we are computing starndar deviation for each column
We should write a module for this

'''

std = S.std(axis=0)  # Standardize data
#print "\nstandard deviation"
#print std

'''
Then we are deviding each row of array from the starndard deviatin array
We should write a module for this
'''
#S /= std

#print "\nData after deviding standard deviation"
#print S
#S += np.random.beta(10,5,size=S.shape)  # Add noise
'''
Then we mix the data by dot producting by arbitary array A

'''

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

#write data
fp = open("data.txt","w")
Data =  X.T
for row in Data:
    for item in row:
        fp.write(str(item)+'\t')
    fp.write('\n')
fp.close()



'''
Then we initialize the object from class fastica
initializing n_components=3

'''
#measuring time
t1 = time.clock()

# Compute ICA
ica = FastICA(n_components=3)

'''
Then we call function: fit_transform by passing matrix X-mixed matrix
'''

S_ = ica.fit_transform(X)  # Reconstruct signals

t2 = time.clock()

print 'ica took ',t2-t1

'''
And we call ica.mixing_  function to get mixing matrix
'''
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

np.savetxt("pythonout.txt",S_,delimiter=' ')

###############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()


