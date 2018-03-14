import math
import numpy as np
import hypothesis
import hypothesis.strategies as st

def check_theta(model, thetas, N=0, check_bounds=True):
    """ 
    Check a single theta or stack of gradient vectors. 
    N=expected number of elements (0=do not check number of elements)
    """
    if N==0 or N==1:
        assert thetas.ndim == 2 or thetas.ndim == 1, "Incorrect theta stack dimension"
    else:
        assert thetas.ndim == 2, "Incorrect theta stack dimension"
    assert thetas.shape[0] == model.num_params, "Incorrect number of params"
    if N==1:
        assert thetas.ndim == 1 or thetas.shape[1] == 1, "Incorrect number of thetas"
    elif N>1:
        assert thetas.shape[1] == N, "Incorrect number of thetas"
    assert np.all(np.isfinite(thetas))
    if check_bounds:
        assert np.all(thetas>0)
        assert np.all(model.theta_in_bounds(thetas))

def check_sample(model, sample, Nsample, N=0):
    """
    Check an mcmc sample or stack of samples
    """
    if N==0 or N==1:
        assert sample.ndim == 3 or sample.ndim == 2, "Incorrect theta stack dimension"
    else:
        assert sample.ndim == 3, "Incorrect theta stack dimension"
    assert sample.shape[0] == model.num_params, "Incorrect number of params"
    assert sample.shape[1] == Nsample, "Incorrect number of samples"
    if N==1:
        assert sample.ndim == 2 or sample.shape[2] == 1, "Incorrect number of sample stacks"
    elif N>1:
        assert sample.shape[2] == N, "Incorrect number of thetas"
    assert np.all(sample>0)
    assert np.all(np.isfinite(sample))
    if N>1:
        for n in range(N):
            assert np.all(model.theta_in_bounds(sample[...,n]))
    
    
def check_llh(llhs, N=0):
    """ 
    Check a single log-likelihood value or vector of values. 
    N=expected number of elements (0=do not check number of elements)
    """
    if N==1:
        assert isinstance(llhs,float) or (isinstance(llhs, np.ndarray) and llhs.size == 1)
    elif N>0:
        assert llhs.ndim == 1
        assert llhs.shape[0] == N
    assert np.all(np.isfinite(llhs))

def check_grad(model, grads, N=0):
    """ 
    Check a single gradient vector or stack of gradient vectors. 
    N=expected number of elements (0=do not check number of elements)
    """    
    if N == 0 or N == 1:
        assert grads.ndim == 2 or grads.ndim == 1, "Incorrect grad stack dimension"
    else:
        assert grads.ndim == 2, "Incorrect grad stack dimension"
    assert grads.shape[0] == model.num_params, "Incorrect gradient shape"
    if N == 1:
        assert grads.ndim == 1 or grads.shape[1] == 1, "Incorrect number of columns"
    elif N>1:
        assert grads.shape[1] == N, "Incorrect number of columns"
    assert np.all(np.isfinite(grads))

def check_symmat(model, mat, N=0):
    """
    Check a single symmetric matrix or stack of symmetric matrices.  
    Hessians and covariance matricies are represented as symmetric full matrices.
    N=expected number of elements (0=do not check number of elements)
    """
    if N == 0 or N == 1:
        assert mat.ndim == 3 or mat.ndim == 2, "Incorrect matrix stack dimension"
    else:
        assert mat.ndim == 3, "Incorrect matrix stack dimension"
    assert mat.shape[0] == model.num_params, "Incorrect matrix shape"
    assert mat.shape[1] == model.num_params, "Incorrect matrix shape"
    if N == 1:
        assert mat.ndim == 2 or mat.shape[2] == 1, "Incorrect number of columns"
    elif N>1:
        assert mat.shape[2] == N, "Incorrect number of columns"
    assert np.all(np.isfinite(mat))
    if mat.ndim == 2:
        assert np.all(mat == np.transpose(mat)), "Matrix should be symmetric"
    else:
        assert np.all(mat == np.transpose(mat,axes=[1,0,2])), "Matrix should be symmetric"

def check_image(model,im, N=0):
    """
    Check a single returned image or stack of returned images is the right shape and value.
    N=expected number of elements (0=do not check number of elements)
    """
    if N == 0 or N == 1:
        assert im.ndim == model.num_dim or im.ndim == model.num_dim+1, "Image stack incorrect ndim"
    else:
        assert im.ndim == model.num_dim+1, "Image stack incorrect ndim"
    if model.num_dim == 1:
        assert im.shape[0] == model.size, "Image stack incorrect base size"
    else:        
        assert np.all(im.shape[1::-1] == model.size), "Image stack incorrect base size"
    if N == 1:
        assert im.ndim == model.num_dim or im.shape[-1] == 1
    elif N>1:
        assert im.shape[-1] == N, "Incorrect number of images"
    assert np.all(np.isfinite(im)), "Images should be finite."
    assert np.all(im >= 0), "Images should be positive."

def check_stats(stats):
    """
    check a returned StatsT stats object is a dictionary mapping strings to floats
    """
    assert isinstance(stats,dict)
    assert all(isinstance(key,str) for key in stats.keys())


def draw_feasible_theta(model, hyp_data):
    """Use a hypothesis.strategies.data() object and lbound,ubound vectors to draw a feasible theta."""
    MAXVAL= 1.0E12;
    theta = np.array([hyp_data.draw(st.floats(L, U, allow_infinity=False)) for L,U in 
                      zip(model.lbound+ model.bounds_epsilon, 
                           [min(MAXVAL,x) for x in model.ubound-model.bounds_epsilon])])
    hypothesis.assume(not np.any(theta == model.lbound+model.bounds_epsilon))
    hypothesis.assume(not np.any(theta == model.ubound-model.bounds_epsilon))
    return theta

def draw_infeasible_theta(model, hyp_data):
    """Use a hypothesis.strategies.data() object and lbound,ubound vectors to draw a feasible theta."""
    theta = np.array([hyp_data.draw(st.floats(L, U, allow_infinity=False)) for L,U in zip(model.lbound, model.ubound)])
    index = hyp_data.draw(st.integers(0,model.num_params-1));
    if math.isinf(model.ubound[index]):
        value = hyp_data.draw(st.floats(-math.inf, model.lbound[index]))
    else:
        value = hyp_data.draw( st.one_of(st.floats(-math.inf, model.lbound[index]),
                                         st.floats(model.ubound[index], math.inf)) )
    theta[index]=value
    return theta


def draw_prior_theta(model, seed, N=0):
    """Use hypotheis-supplied rng seed and model.sample_prior() to draw a theta from the prior distribution."""
    model.set_rng_seed(seed)
    if N==0:
        theta = model.sample_prior()
        check_theta(model,theta,1)
        return theta
    else:
        thetas = model.sample_prior(N)
        check_theta(model,thetas,N)
        return thetas
    
