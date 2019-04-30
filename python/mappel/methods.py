# mappel/methods.py
#
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# Additional method definitions that will be added to all Mappel models.
# function _WrapModelClass(model_class) controls the addition of extra methods.
#
import numpy as np

from . common import *
from . process_stats import process_stats, process_estimator_debug_stats

def _estimate_max_wrapper(self,image,method=DefaultEstimatorMethod,theta_init=np.empty((0,),dtype='double'),return_stats=False):
    """
    Returns (theta_max_stack,rllh_stack,observedI_stack,stats). Estimates the maximum of the model objective.  
        
    This is Maximum likelihood estimation (MLE) or maximum-aposeteriori (MAP) estimation depending on the model.  
    fixed_theta is a vector of fixed values free values are indicated by inf or nan. [OpenMP]
    """
    if not return_stats:
        return self._estimate_max(image,method,theta_init,return_stats)
    else:
        (theta_est, rllh, obsI, raw_stats) = self._estimate_max(image,method,theta_init,return_stats)
        stats = process_stats(raw_stats)
        return (theta_est, rllh, obsI, stats)


def _estimate_profile_likelihood_wrapper(self,image,fixed_parameter_idxs, fixed_parameter_values, method=DefaultEstimatorMethod,theta_init=np.empty((0,),dtype='double'),return_stats=False):
    """
    Returns (profile_likelihood, profile_parameters, stats).  Estimate the profile likelihood for a single image, given fixed_parameter_idxs and fixed_values.

    Fixed parameter indexes has the 0-based index of each fixed parameter.  Each column of fixed_parameter_values represents a fixed value to
    use for the fixed parameters while maximizing over the remaining free parameters.  Uses OpenMP to evaluate values in parallel.
    """
    if not return_stats:
        return self._estimate_profile_likelihood(image,fixed_parameters_idxs, fixed_parameter_values, method, theta_init, return_stats)
    else:
        (profile_likelihood, profile_parameters, raw_stats) = self._estimate_profile_likelihood(image,fixed_parameters_idxs, fixed_parameter_values, method, theta_init, return_stats)
        stats = process_stats(raw_stats)
        return (profile_likelihood, profile_parameters, stats)

#def _estimate_max_debug_wrapper(self,image,method=DefaultEstimatorMethod,theta_init=np.empty((0,),dtype='double')):
    #"""
    #[Debugging Usage] Returns (theta_est, rllh, obsI, seq, seq_rllh, stats).
    #"""
    #(theta_est, rllh, obsI, seq, seq_rllh, raw_stats) = self._estimate_max_debug(image,method,theta_init)
    #stats = process_estimator_debug_stats(raw_stats)
    #return (theta_est, rllh, obsI, seq, seq_rllh, stats)

def _error_bounds_profile_likelihood_wrapper(self,images,theta_mle, confidence=DefaultConfidenceLevel,theta_mle_rllh=float("-inf"),obsI=np.empty((0,),dtype='double'),estimated_idxs=np.empty((0,),dtype='uint64'),return_stats=False):
    """
     Returns (profile_lb, profile_ub, profile_points_lb, profile_points_ub, profile_points_lb_rllh, profile_points_ub_rllh, stats).

    Error bounds for each estimated parameter.  Profile points are the points where the bounds were found.  Uses the Venzon and Moolgavkar 1988 algorithm.
    The profile likelihood error estimation makes no assumptions about the Normality of the likelihood distribution near the MLE and is applicable in
    interval estimation for non-regular objective functions.  It is a pure-likelihood based approach to find the estimated error bounds.
    [OpenMP over dimensions or images]
    """
    if not return_stats:
        return self._error_bounds_profile_likelihood(images,theta_mle, confidence, theta_mle_rllh, obsI, estimated_idxs, return_stats)
    else:
        (profile_lb, profile_ub, profile_points_lb, profile_points_ub, profile_points_lb_rllh, profile_points_ub_rllh, raw_stats) = self._error_bounds_profile_likelihood(images,theta_mle, confidence, theta_mle_rllh, obsI, estimated_idxs, return_stats)
        stats = process_stats(raw_stats)
        return (profile_lb, profile_ub, profile_points_lb, profile_points_ub, profile_points_lb_rllh, profile_points_ub_rllh, stats)

def _get_stats_wrapper(self):
    """
    Get a dictionary of model settings.
    """
    return process_stats(self._get_stats())


def _WrapModelClass(model_class):
    """
    Modify a Mappel model class type, adding in extra methods and wrappers.
    This is applied by __init__.py to all imported modules
    """
    model_class.get_stats = _get_stats_wrapper
    model_class.estimate_max = _estimate_max_wrapper
    model_class.estimate_profile_likelihood = _estimate_profile_likelihood_wrapper
    #model_class.estimate_max_debug = _estimate_max_debug_wrapper
