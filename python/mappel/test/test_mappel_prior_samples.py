# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# These tests use hypothesis to (repeatably) generate random seeds, with
# which the random number generator is seeded, and a prior samle of size Nstack is
# drawn to test with.  These methods test all of the OpenMP operations of the numerical
# methods.
#
#

import numpy as np
import scipy.linalg
import pytest
import hypothesis
#import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from hypothesis import settings
from .common import *
settings.register_profile("prior_samples", max_examples=50)
settings.load_profile("prior_samples")

Nstack = 80 #Number of samples for to test a vector operation with

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_sample_prior_single(model, seed):
    """Check model.sample_prior and model.set_rng_seed repeatability"""
    theta = draw_prior_theta(model,seed)
    theta2 = draw_prior_theta(model,seed)
    assert np.all(theta == theta2), "Samples drawn with same seed should be identical"
    theta3 = model.sample_prior()
    assert np.any(theta2 != theta3), "Successive samples should differ."

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_sample_prior(model, seed):
    """Check model.sample_prior and model.set_rng_seed repeatability"""
    theta = draw_prior_theta(model,seed,Nstack)
    theta2 = draw_prior_theta(model,seed,Nstack)
    assert np.all(theta == theta2), "Samples drawn with same seed should be identical"
    theta3 = model.sample_prior(Nstack)
    assert np.any(theta2 != theta3), "Successive samples should differ."

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_model_image(model, seed):
    """Check the model.model_image() image for sanity."""
    theta = draw_prior_theta(model,seed,Nstack)
    im = model.model_image(theta)
    check_image(model,im,Nstack)
    assert np.all(im == model.model_image(theta)), "Model image should be repeatable"
    
@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_simulate_image(model, seed):
    """Check the model.simulate_image() for sanity."""
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta,Nstack)
    check_image(model,im,Nstack)


## Objective ##

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective_llh(model,seed):
    """Check model.objective_llh vectorized calls work for all argument patterns."""
    theta = draw_prior_theta(model,seed, Nstack)
    im = model.simulate_image(theta)
    #four call patterns for Nims,Nthetas
    llh_nn = model.objective_llh(im,theta) #N images, N thetas
    llh_1n = model.objective_llh(im[...,0],theta) # one image, N thetas
    llh_n1 = model.objective_llh(im,theta[:,0])   # N images, one theta
    llh_11 = model.objective_llh(im[...,0],theta[:,0]) # one image, one theta

    assert np.all(np.isfinite(llh_nn))
    assert np.all(np.isfinite(llh_1n))
    assert np.all(np.isfinite(llh_n1))
    assert np.all(np.isfinite(llh_11))
    assert llh_nn.size == Nstack
    assert llh_n1.size == Nstack
    assert llh_1n.size == Nstack
    assert llh_11.size == 1
    assert llh_nn[0] == llh_1n[0]
    assert llh_nn[0] == llh_n1[0]
    assert llh_nn[0] == llh_11[0]


@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective_rllh(model,seed):
    """Check model.objective_rllh vectorized calls work for all argument patterns."""
    theta = draw_prior_theta(model,seed, Nstack)
    im = model.simulate_image(theta)
    #four call patterns for Nims,Nthetas
    rllh_nn = model.objective_rllh(im,theta) #N images, N thetas
    rllh_1n = model.objective_rllh(im[...,0],theta) # one image, N thetas
    rllh_n1 = model.objective_rllh(im,theta[:,0])   # N images, one theta
    rllh_11 = model.objective_rllh(im[...,0],theta[:,0]) # one image, one theta

    assert np.all(np.isfinite(rllh_nn))
    assert np.all(np.isfinite(rllh_1n))
    assert np.all(np.isfinite(rllh_n1))
    assert np.all(np.isfinite(rllh_11))
    assert rllh_nn.size == Nstack
    assert rllh_n1.size == Nstack
    assert rllh_1n.size == Nstack
    assert rllh_11.size == 1
    assert rllh_nn[0] == rllh_1n[0]
    assert rllh_nn[0] == rllh_n1[0]
    assert rllh_nn[0] == rllh_11[0]

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective_grad(model,seed):
    """Check model.objective_grad vectorized calls work for all argument patterns."""
    theta = draw_prior_theta(model,seed, Nstack)
    im = model.simulate_image(theta)
    #four call patterns for Nims,Nthetas
    grad_nn = model.objective_grad(im,theta) #N images, N thetas
    grad_1n = model.objective_grad(im[...,0],theta) # one image, N thetas
    grad_n1 = model.objective_grad(im,theta[:,0])   # N images, one theta
    grad_11 = model.objective_grad(im[...,0],theta[:,0]) # one image, one theta
    check_grad(model, grad_nn, Nstack)
    check_grad(model, grad_n1, Nstack)
    check_grad(model, grad_1n, Nstack)
    check_grad(model, grad_11, 1)
    assert np.all(grad_nn[...,0] == grad_1n[...,0])
    assert np.all(grad_nn[...,0] == grad_n1[...,0])
    assert np.all(grad_nn[...,0] == grad_11)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective_hessian(model,seed):
    """Check model.objective_hessian vectorized calls work for all argument patterns."""
    theta = draw_prior_theta(model,seed, Nstack)
    im = model.simulate_image(theta)
    #four call patterns for Nims,Nthetas
    hess_nn = model.objective_hessian(im,theta) #N images, N thetas
    hess_1n = model.objective_hessian(im[...,0],theta) # one image, N thetas
    hess_n1 = model.objective_hessian(im,theta[:,0])   # N images, one theta
    hess_11 = model.objective_hessian(im[...,0],theta[:,0]) # one image, one theta
    check_symmat(model, hess_nn, Nstack)
    check_symmat(model, hess_n1, Nstack)
    check_symmat(model, hess_1n, Nstack)
    check_symmat(model, hess_11, 1)
    assert np.all(hess_nn[...,0] == hess_1n[...,0])
    assert np.all(hess_nn[...,0] == hess_n1[...,0])
    assert np.all(hess_nn[...,0] == hess_11)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective_negative_definite_hessian(model,seed):
    """Check model.objective_negative_definite_hessian vectorized calls work for all argument patterns."""
    theta = draw_prior_theta(model,seed, Nstack)
    im = model.simulate_image(theta)
    #four call patterns for Nims,Nthetas
    hess_nn = model.objective_negative_definite_hessian(im,theta) #N images, N thetas
    hess_1n = model.objective_negative_definite_hessian(im[...,0],theta) # one image, N thetas
    hess_n1 = model.objective_negative_definite_hessian(im,theta[:,0])   # N images, one theta
    hess_11 = model.objective_negative_definite_hessian(im[...,0],theta[:,0]) # one image, one theta
    check_symmat(model, hess_nn, Nstack)
    check_symmat(model, hess_n1, Nstack)
    check_symmat(model, hess_1n, Nstack)
    check_symmat(model, hess_11, 1)
    assert np.all(hess_nn[...,0] == hess_1n[...,0])
    assert np.all(hess_nn[...,0] == hess_n1[...,0])
    assert np.all(hess_nn[...,0] == hess_11)
    #This will throw an scipy.LinAlgError if the hessians are not negatiave definite
    chol = [ scipy.linalg.cholesky(-hess_nn[...,n]) for n in range(Nstack)]
    
@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_objective(model,seed):
    """Check the model.objective() return values against direct computation."""
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    val = model.objective(im,theta)
    assert len(val) == 3, "Should return 3 components"
    (rllh,grad,hess) = val
    check_grad(model,grad)
    check_symmat(model,hess)
    assert rllh == model.objective_rllh(im,theta)
    assert np.all(grad == np.squeeze(model.objective_grad(im,theta)))
    assert np.all(hess == np.squeeze(model.objective_hessian(im,theta)))

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_likelihood_objective(model,seed):
    """Check the model.likelihood_objective() for return value sanity."""
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    val = model.likelihood_objective(im,theta)
    assert len(val) == 3, "Should return 3 components"
    (rllh,grad,hess) = val
    assert math.isfinite(rllh)
    check_grad(model,grad)
    check_symmat(model,hess)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_prior_objective(model,seed):
    """Check the model.prior_objective() for return value sanity."""
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    val = model.prior_objective(theta)
    assert len(val) == 3, "Should return 3 components"
    (rllh,grad,hess) = val
    assert math.isfinite(rllh)
    check_grad(model,grad)
    check_symmat(model,hess)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_aposteriori_objective(model,seed):
    """Check the model.prior_objective() for return value sanity."""
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    val = model.aposteriori_objective(im,theta)
    assert len(val) == 3, "Should return 3 components"
    (rllh,grad,hess) = val
    assert math.isfinite(rllh)
    check_grad(model,grad)
    check_symmat(model,hess)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_compare_objectives(model,seed):
    """
    Check the model.likelihood_objective() model.asposteriori_objective() and 
    model.prior_objective()return values against direct computation.
    """
    theta = draw_prior_theta(model,seed)
    im = model.simulate_image(theta)
    likelihood_val = model.likelihood_objective(im,theta)
    prior_val = model.prior_objective(theta)
    aposteriori_val = model.aposteriori_objective(im,theta)
    assert aposteriori_val[0] == likelihood_val[0] + prior_val[0], "Objective rllh consistency"
    assert np.all(aposteriori_val[1] == likelihood_val[1] + prior_val[1]), "Objective grad consistency"
    assert np.all(aposteriori_val[2] == likelihood_val[2] + prior_val[2]), "Objective hess consistency"
    
@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_cr_lower_bound(model,seed):
    """Check the model.cr_lower_bound() for return value sanity."""
    theta = draw_prior_theta(model,seed, Nstack)
    crlb = model.cr_lower_bound(theta)
    assert np.all(np.isfinite(crlb))
    assert np.all(crlb>=0.)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_expected_information(model,seed):
    """Check the model.expected_information() for return value sanity."""
    theta = draw_prior_theta(model,seed, Nstack)
    fisherI = model.expected_information(theta)
    check_symmat(model,fisherI,Nstack) # fisherI should act like a hessian matrix (or stack of hessians)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_observed_information(model,seed):
    """
    Check the model.observed_information() for return value sanity.
    This only checks the input/output functionality, as we test-evaluate observed information 
    not at the maximum, but at some sampled thetas and simulated images. 
    """
    theta = draw_prior_theta(model,seed,1)
    im = model.simulate_image(theta)
    obsI = model.observed_information(im,theta)
    check_symmat(model,obsI,1) # fisherI should act like a hessian matrix (or stack of hessians)

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
def test_error_bounds_expected(model,seed):
    """
    Check model.error_bounds_expected()
    """
    theta = draw_prior_theta(model,seed, Nstack)
    val = model.error_bounds_expected(theta)
    assert len(val) == 2
    (lbound,ubound) = val
    check_theta(model,lbound,Nstack,False)
    check_theta(model,ubound,Nstack,False)
    assert np.all(lbound < theta)
    assert np.all(ubound > theta)
