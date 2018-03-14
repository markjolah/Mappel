# test_mappel_2DFixedSigma.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 2D fixed sigma model tests

import numpy as np
        
def test_psf_sigma_2D(model2DFixedSigma):
    """Check psf_sigma get and set properties."""
    model = model2DFixedSigma
    assert np.all(model.psf_sigma >= model.global_min_psf_sigma)
    psf_sigma = model.psf_sigma
    for n in range(model.num_dim):
        psf_sigma[n] += 0.2;
        model.psf_sigma = psf_sigma
        assert np.all(model.psf_sigma == psf_sigma)
        psf_sigma[n] -= 0.2;
        
