# histogram_plots.py
# working script for generating 1-D plots comparing Centroid to Least Squares to MLE

import numpy as np
#import scipy.special as sp
import mappel
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import mappel.view

#M = mappel.Gauss1DMLE(8,1.0)
#P_samp = [3.5,10000,20]
#im = M.simulate_image(P_samp)

A = mappel.view.viewer1D()
A.overlayModelSim([2,10000,50])
A.overlayModelSim([2,1000,100])
A.overlayModelSim([2,100,100])

