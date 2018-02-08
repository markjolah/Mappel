import numpy as np
import scipy.special as sp
import mappel
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-ticks')

## 1D histogram, sample data and model overlay
M = mappel.Gauss1DMLE(8,1.0)
P_samp = M.sample_prior(1)
psf_var = M.psf_sigma**2
im = M.simulate_image(P_samp)
md = M.model_image(P_samp)
P_out, LLH, Hess = M.estimate_max(im,'Newton',P_samp)
Normerf = 0.5*(sp.erf( (np.ceil(P_out[0])-P_out[0])/2/psf_var ) - sp.erf( (np.floor(P_out[0])-P_out[0])/2/psf_var ) )

# plot simulated data, expected model, and MLE estimate
fig, ax = plt.subplots()
fig_im = ax.bar(range(im.size),im,color='r',label='simulated')
fig_md = ax.bar(range(md.size),md,color='b',alpha=0.4,label='model')
fig_MLE = ax.plot(P_out[0]-0.5,P_out[1]*Normerf+P_out[2],'k+',markersize=12,label='MLE Estimate')
ax.set_title('Simulated and Model histogram with MLE marker')
ax.set_xlabel('pixel position')
ax.set_ylabel('pixel count')
ax.legend()

plt.show(fig)
