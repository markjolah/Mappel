# test_mappel_2DVariableSigma.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 2D variable sigma model tests
import math
import numpy as np

def test_min_max_sigma_2D(model2DVariableSigma):
    """Check min_sigma and max_sigma get and set properties."""
    model = model2DVariableSigma
    assert np.all(model.global_min_psf_sigma <= model.min_sigma) 
    assert np.all(model.min_sigma < model.max_sigma)
    assert np.all(model.max_sigma <= model.global_max_psf_sigma)
    assert np.all(np.isfinite(model.min_sigma))
    assert np.all(np.isfinite(model.max_sigma))
    #check setter properties
    for n in range(model.num_dim):
        min_sigma = model.min_sigma
        max_sigma = model.max_sigma
        min_sigma[n] /= 2
        max_sigma = min_sigma*3.1;
        model.min_sigma = min_sigma;
        model.max_sigma = max_sigma;
        assert np.all(model.min_sigma == min_sigma)
        assert np.all(model.max_sigma == max_sigma)
        assert model.max_sigma_ratio == 3.1

