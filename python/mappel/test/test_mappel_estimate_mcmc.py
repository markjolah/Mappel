# test_mappel_estimate_mcmc.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# Test the model.estimate_mcmc related methods
#
#
import numpy as np
import pytest
import hypothesis
import hypothesis.extra.numpy as npst
from hypothesis import settings

from .conftest import MappelEstimatorTestMethods
from .common import *

settings.register_profile("estimate_mcmc", max_examples=5)
settings.load_profile("estimate_mcmc")
Nestimate = 32
Nsample = 200
Nburnin = 30
thin = 0

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_estimate_mcmc_sample(model,seed):
    """ Check model.estimate_mcmc_sample() """
    theta = draw_prior_theta(model,seed,Nestimate)
    im = model.simulate_image(theta)
    model.set_rng_seed(seed) #re-seed for repeatability
    val = model.estimate_mcmc_sample(images=im, Nsample=Nsample, Nburnin=Nburnin, thin=thin)
    assert len(val) == 2
    (sample, sample_rllh) = val
    check_sample(model, sample, Nsample, Nestimate)
    for n in range(Nestimate):
        check_llh(sample_rllh[...,n], Nsample)
        
    model.set_rng_seed(seed) #re-seed for repeatability
    val2 = model.estimate_mcmc_sample(images=im, Nsample=Nsample, Nburnin=Nburnin, thin=thin)
    assert len(val2) == 2
    (sample2, sample_rllh2) = val2
    assert np.all(sample2==sample)
    

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_estimate_mcmc_posterior(model,seed):
    """ Check model.estimate_mcmc_sample() """
    theta = draw_prior_theta(model,seed,Nestimate)
    im = model.simulate_image(theta)
    model.set_rng_seed(seed) #re-seed for repeatability
    val = model.estimate_mcmc_posterior(images=im, Nsample=Nsample, Nburnin=Nburnin, thin=thin)
    assert len(val) == 2
    (theta_mean, theta_cov) = val
    check_theta(model, theta_mean, Nestimate)
    check_symmat(model, theta_cov, Nestimate)    

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_estimate_mcmc_debug(model,seed):
    """ Check model.estimate_mcmc_debug() """
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    model.set_rng_seed(seed) #re-seed for repeatability
    theta_init = model.initial_theta_estimate(im)
    val = model.estimate_mcmc_debug(im, Nsample=Nsample, theta_init=theta_init)
    assert len(val) == 4
    (sample, sample_rllh, candidate, candidate_rllh) = val
    check_sample(model, sample, Nsample, 1)
    check_llh(sample_rllh, Nsample)
    check_theta(model,sample,Nsample,False)
    assert candidate_rllh.ndim == 1
    assert candidate_rllh.shape[0] == Nsample
    for n in range(Nsample):
        if model.theta_in_bounds(candidate[:,n]):
            assert candidate_rllh[n] == model.objective_rllh(im,candidate[:,n]), "Inbounds point rllh does not match."
        else:
            assert candidate_rllh[n] == -math.inf, "OOB point should have -inf as llh"
    
    assert np.all(sample[:,0] == theta_init), "Theta init is not first sample."
        
    model.set_rng_seed(seed) #re-seed for repeatability
    val2 = model.estimate_mcmc_sample(images=im, theta_init=theta_init, Nsample=Nsample, Nburnin=0, thin=1)
    assert len(val2) == 2
    (sample2, sample_rllh2) = val2
    assert np.all(sample2 == sample), "Debug output does not match regular."
    assert np.all(sample_rllh2 == sample_rllh), "Debug output does not match regular."

#@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
#def test_error_bounds_posterior_credible(model,seed):
    #"""
    #Check model.error_bounds_posterior_credible()
    #"""
    #theta = draw_prior_theta(model,seed, Nestimate)
    #im = model.simulate_image(theta)
    #(sample, sample_rllh) = model.estimate_mcmc_sample(im,Nsample=Nsample,Nburnin=Nburnin,thin=thin)
    #check_sample(model, sample, Nsample, Nestimate)
    #val = model.error_bounds_posterior_credible(sample)
    #assert len(val) == 3
    #(theta_mean,lbound,ubound) = val
    #check_theta(model,theta_mean,Nestimate,check_bounds=True)
    #check_theta(model,lbound,Nestimate,check_bounds=True)
    #check_theta(model,ubound,Nestimate,check_bounds=True)
    #assert np.all(lbound < theta_mean)
    #assert np.all(ubound > theta_mean)
    
