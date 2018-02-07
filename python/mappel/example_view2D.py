import numpy as np
import mappel
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-ticks')

## 2D histogram, sample data and model side by side

M = mappel.Gauss2DMLE([8,8],[1.0,1.0])
P_samp = M.sample_prior(1)
im = M.simulate_image(P_samp)
md = M.model_image(P_samp)
P_out, LLH, Hess = M.estimate_max(im,'Newton',P_samp)

# plot simulated data, expected model, and MLE estimate
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
fig_im = ax0.imshow(np.squeeze(im))
fig_md = ax1.imshow(np.squeeze(md))
MLE0 = ax0.plot(P_out[0]-0.5,P_out[1]-0.5,'ro')
MLE1 = ax1.plot(P_out[0]-0.5,P_out[1]-0.5,'ro')
ax0.set_title('Simulated vs Model image overlaid with MLE')

plt.show(fig)
