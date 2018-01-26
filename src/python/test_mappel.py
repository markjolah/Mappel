import numpy as np
import mappel

M=mappel.Gauss1DMLE(8,1);
n=10;

theta = M.sample_prior(n);
ims = M.simulate_image(theta);

llh = M.objective_llh(ims,theta);
