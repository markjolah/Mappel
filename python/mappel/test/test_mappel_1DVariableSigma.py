# test_mappel_1DVariableSigma.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 1D variable sigma model tests
import math

def test_min_max_sigma_1D(model1DVariableSigma):
    """Check min_sigma and max_sigma get and set properties."""
    model = model1DVariableSigma
    assert 0 < model.global_min_psf_sigma 
    assert model.global_min_psf_sigma <=model.min_sigma 
    assert model.min_sigma < model.max_sigma
    assert model.max_sigma <= model.global_max_psf_sigma
    assert math.isfinite(model.min_sigma)
    assert math.isfinite(model.max_sigma)
    #check setter properties
    min_sigma = model.min_sigma
    max_sigma = model.max_sigma
    model.min_sigma *= 0.9
    model.max_sigma *= 1.1
    assert model.min_sigma == min_sigma*0.9
    assert model.max_sigma == max_sigma*1.1

def test_min_max_sigma_bounds_1D(model1DVariableSigma):
    """Check min_sigma and max_sigma are respected in bounds."""
    model = model1DVariableSigma
    assert model.min_sigma == model.lbound[-1]
    assert model.max_sigma == model.ubound[-1]

    lbound = model.lbound
    new_lbound = 0.8
    lbound[-1] = new_lbound
    model.lbound = lbound
    assert model.lbound[-1] == new_lbound
    assert model.min_sigma == new_lbound
    
    ubound = model.ubound
    new_ubound = 2.8
    ubound[-1] = new_ubound
    model.ubound = ubound
    assert model.ubound[-1] == new_ubound
    assert model.max_sigma == new_ubound
