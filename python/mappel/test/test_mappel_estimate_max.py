# test_mappel_estimate_max.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# Test the model.estimate_max() and related methods
#
#
import numpy as np
import scipy.linalg
import pytest
import hypothesis
import hypothesis.extra.numpy as npst
from hypothesis import settings

from .conftest import MappelEstimatorTestMethods
from .common import *

settings.register_profile("estimate_max", max_examples=5)
settings.load_profile("estimate_max")
Nestimate = 32

@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
@pytest.mark.parametrize("method",MappelEstimatorTestMethods)
def test_estimate_max(model,method,seed):
    """
    Check model.estimate_max() Over a pararameterized list of estimator names.
    Checks estimation is repeatable (reseeds RNG for simulatedannealing method)
    """
    theta = draw_prior_theta(model,seed,Nestimate)
    im = model.simulate_image(theta)
    model.set_rng_seed(seed) #re-seed for repeatability
    val = model.estimate_max(im,method,return_stats=True)
    assert len(val) == 4
    (theta_est, rllh, obsI, stats) = val
    check_theta(model,theta_est,Nestimate)
    check_llh(rllh,Nestimate)
    check_symmat(model,obsI,Nestimate)
    check_stats(stats)

    #check repeatability
    model.set_rng_seed(seed) #re-seed for repeatability
    val2 = model.estimate_max(im,method,return_stats=False)
    assert len(val2) == 3
    (theta_est2, rllh2, obsI2) = val2
    assert np.all(theta_est == theta_est2)
    assert np.all(rllh == rllh2)
    assert np.all(obsI == obsI2)
    
#@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
#@pytest.mark.parametrize("method",MappelEstimatorTestMethods)
#def test_estimate_max_debug(model,method,seed):
    #"""
    #Check model.estimate_max_debug()
    #"""
    #theta = draw_prior_theta(model,seed)
    #im = model.simulate_image(theta)
    #model.set_rng_seed(seed) #re-seed for repeatability
    #theta_init = model.initial_theta_estimate(im)
    #check_theta(model,theta_init,1)
    #val = model.estimate_max_debug(im,method,theta_init)
    #assert len(val) == 6
    #(theta_est, rllh, obsI, stats, sequence, sequence_rllh) = val
    #check_theta(model,theta_est,1)
    #check_llh(rllh,1)
    #check_symmat(model,obsI,1)
    #check_stats(stats)
    #check_theta(model,sequence)
    #check_llh(sequence_rllh)
    #assert np.all(theta_init == sequence[:,0]), "theta_init should be first element in sequence"
    #if "backtrack_idxs" in stats:
        ##theta_est should be last non backtrack in sequence
        #backtrack_idxs = stats["backtrack_idxs"]
        #assert backtrack_idxs.size == sequence.shape[1], "Backtrack indexs do not match sequence length"
        #non_backtrack_idxs = np.where(~backtrack_idxs)[0]
        #last_non_backtrack_idx = non_backtrack_idxs[-1]  #Last non-backtracking evaluation (last accepted)
        #assert np.all(theta_est == sequence[:,last_non_backtrack_idx]), "theta_est should be last non-backtrack theta in sequence"
        #assert rllh == sequence_rllh[last_non_backtrack_idx], "rllh should match last non-backtrack rllh in sequence_rllh"
    #else:
        ##theta_est should be last in sequence
        #assert np.all(theta_est == sequence[:,-1]), "theta_est should be last element in sequence"
    #rllh2 = model.objective_rllh(im,sequence)
    #assert np.all(rllh2 == sequence_rllh), "sequence_rllh does not match with separate computation"
    
    ##check estimate is the same as in regular estimate_max()
    #model.set_rng_seed(seed) #re-seed for repeatability
    #val2 = model.estimate_max(im,method,theta_init)
    #assert len(val2) == 3
    #(theta_est2,rllh2,obsI2) = val2
    #assert np.all(theta_est == theta_est2), "theta_est does not match non-debug"
    #assert np.all(obsI == obsI2), "obsI does not match non-debug"
    #assert np.all(rllh == rllh2), "rllh does not match non-debug"
    
    
    
#@hypothesis.given(seed=npst.arrays(shape=1,dtype="uint64"))
#def test_error_bounds_observed(model,seed):
    #"""
    #Check model.error_bounds_observed()
    #"""
    #theta = draw_prior_theta(model,seed, Nestimate)
    #im = model.simulate_image(theta)
    #(theta_est, rllh, obsI) = model.estimate_max(im,"NewtonDiagonal")
    ##This will throw an scipy.LinAlgError if the hessians are not negatiave definite
    #chol = [ scipy.linalg.cholesky(-obsI[...,n]) for n in range(Nestimate)]
    #val = model.error_bounds_observed(theta_est,obsI)
    #assert len(val) == 2
    #(lbound,ubound) = val
    #check_theta(model,lbound,Nestimate,False)
    #check_theta(model,ubound,Nestimate,False)
    #assert np.all(lbound < theta_est)
    #assert np.all(ubound > theta_est)

    
    
    
    
