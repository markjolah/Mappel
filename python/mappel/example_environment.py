import numpy as np
import mappel
psf_sigma =np.array([0.9,1.2],dtype='double');
max_sigma_ratio = 3.0
size = [8,10];
m1D = mappel.Gauss1DMAP(size[0], psf_sigma[0]);
m1Ds = mappel.Gauss1DsMAP(size[0], psf_sigma[0], max_sigma_ratio*psf_sigma[0]);
m2D = mappel.Gauss2DMAP(size, psf_sigma);
m2Ds = mappel.Gauss2DsMAP(size, psf_sigma, max_sigma_ratio*psf_sigma);

theta = m1D.sample_prior();
im = m1D.simulate_image(theta);
N=400;
seed =21;
m1D.set_rng_seed(seed);
S1 = m1D.estimate_mcmc_sample(im,N,Nburnin=0,thin=1);
m1D.set_rng_seed(seed);
S2 = m1D.estimate_mcmc_debug(im,N);




