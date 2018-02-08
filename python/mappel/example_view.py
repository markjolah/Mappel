import numpy as np
import scipy.special as sp
import mappel
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-ticks')

## 1D histogram, sample data and model overlay
M = mappel.Gauss1DMLE(8,1.0)
P_samp = M.sample_prior(1)
psf_sig = M.psf_sigma
im = M.simulate_image(P_samp)
md = M.model_image(P_samp)
P_out, LLH, Hess = M.estimate_max(im,'Newton',P_samp)
Normerf = 0.5*(sp.erf( (np.ceil(P_out[0])-P_out[0])/np.sqrt(2)/psf_sig ) - sp.erf( (np.floor(P_out[0])-P_out[0])/np.sqrt(2)/psf_sig ) )
# generate theoretical curve
sampling_freq = 100
gauss_simulated = np.zeros(im.size*sampling_freq)
gauss_model = np.zeros(im.size*sampling_freq)
normfunc = lambda x,x0,I0,bg0,sig: I0/np.sqrt(2*np.pi) * np.exp(-(x-x0)**2/2/sig**2) + bg0
for ii in range(gauss_simulated.size):
    gauss_simulated[ii] = normfunc(ii/sampling_freq,P_out[0],P_out[1],P_out[2],psf_sig)
    gauss_model[ii] = normfunc(ii/sampling_freq,P_samp[0],P_samp[1],P_samp[2],psf_sig)

# plot simulated data, expected model, and MLE estimate
fig, ax = plt.subplots()
fig_im = ax.bar(range(im.size),im,color='r',label='simulated')
fig_md = ax.bar(range(md.size),md,color='b',alpha=0.4,label='model')
fig_imgauss = ax.plot(np.arange(gauss_simulated.size)/sampling_freq-0.5,gauss_simulated,label='MLE Gaussian from Simulation')
fig_mdgauss = ax.plot(np.arange(gauss_model.size)/sampling_freq-0.5,gauss_model,label='Gaussian from Prior Samples')
fig_MLE = ax.plot(P_out[0]-0.5,P_out[1]*Normerf+P_out[2],'k+',markersize=12,label='MLE Estimate')
ax.set_title('Simulated and Model histogram with MLE marker')
ax.set_xlabel('pixel position')
ax.set_ylabel('pixel count')
ax.legend()

#print(P_out)
#print(P_samp)

plt.show(fig)
