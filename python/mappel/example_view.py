import numpy as np
import mappel
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-ticks')

## 1D histogram, sample data and model overlay
M = mappel.Gauss1DMLE(8,1.0)
P_samp = M.sample_prior(1)
im = M.simulate_image(P_samp)
md = M.model_image(P_samp)
P_out, LLH, Hess = M.estimate_max(im,'Newton',P_samp)

# plot simulated data, expected model, and MLE estimate
fig, ax = plt.subplots()
fig_im = ax.bar(range(im.size),im,color='r')
fig_md = ax.bar(range(md.size),md,color='b')
MLE = ax.plot(P_out[0]-0.5,P_out[1],'k+')
ax.set_title('Simulated and Model histogram with MLE marker')

plt.show(fig)
