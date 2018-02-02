import numpy as np
import mappel
M=mappel.Gauss1DMLE(8,1)
N=10
thetas = M.sample_prior(N)
ims = M.simulate_image(thetas)

theta = M.sample_prior()
im = M.simulate_image(theta)




