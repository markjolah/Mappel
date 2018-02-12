import numpy as np
import mappel
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#plt.style.use('seaborn-ticks')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size' : 22}
mpl.rc('font', **font)

## 2D histogram, sample data and model side by side
M = mappel.Gauss2DMLE([8,8],[1.0,1.0])
#P_samp = M.sample_prior(1)
#P_samp=[3.1,4.3,500,50]
P_samp=[3.5,3.5,50,10]
im = M.simulate_image(P_samp)
md = M.model_image(P_samp)
P_out, LLH, Hess = M.estimate_max(im,'Newton',P_samp)
mle_im = M.model_image(P_out)

# plot simulated data, expected model, and MLE estimate
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
fig_im = ax0.imshow(np.squeeze(im))
fig_mle = ax1.imshow(np.squeeze(mle_im))
true0 = ax0.plot(P_samp[0]-0.5,P_samp[1]-0.5,'c+',ms=15,mew=6,label='truth')
true1 = ax1.plot(P_samp[0]-0.5,P_samp[1]-0.5,'c+',ms=15,mew=6,label='truth')
MLE0 = ax0.plot(P_out[0]-0.5,P_out[1]-0.5,'rx',ms=15,mew=6,label='MLE')
MLE1 = ax1.plot(P_out[0]-0.5,P_out[1]-0.5,'rx',ms=15,mew=6,label='MLE')
ax0.legend()
ax1.legend()
ax0.set_title('Simulated vs Estimated Model image')

plt.show(fig)

#fig, (ax0, ax1) = plt.subplots(1,2)
#fig_im = ax1.imshow(np.squeeze(im))
#fig_md = ax0.imshow(np.squeeze(md))
#ax0.set_title('Model to Image Realization')

#plt.show(fig)
