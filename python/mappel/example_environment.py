import numpy as np
import mappel
psf_sigma =np.array([0.9,1.2],dtype='double');
max_sigma_ratio = 3.0
size = [8,10];
m1D = mappel.Gauss1DMAP(size[0], psf_sigma[0]);
m1Ds = mappel.Gauss1DsMAP(size[0], psf_sigma[0], max_sigma_ratio*psf_sigma[0]);
m2D = mappel.Gauss2DMAP(size, psf_sigma);
m2Ds = mappel.Gauss2DsMAP(size, psf_sigma, max_sigma_ratio*psf_sigma);




