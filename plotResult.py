"""
=====================================
Plotting data
=====================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt


#load text to array

data = np.loadtxt("result.txt")
X = data
print data

###############################################################################
# Plot results

plt.figure()

models = [X]
names = ['ICA recovered signals']
colors = ['red', 'steelblue', 'orange','black','green','gray','gold','coral','plum','crimson']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
