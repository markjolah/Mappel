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
    
def _estimate_max_debug_wrapper(self,image,method=DefaultEstimatorMethod,theta_init=np.empty((0,),dtype='double')):
    """
    [Debugging Usage] Returns (sample, sample_rllh, candidates, candidates_rllh).  
    Running MCMC sampling for a single image.  No thinning or burnin is performed. Candidates are the 
    proposed theta values at each iteration.
    """
    (theta_est, rllh, obsI, raw_stats, seq, seq_rllh) = self._estimate_max_debug(image,method,theta_init)
    stats = process_estimator_debug_stats(raw_stats)
    return (theta_est, rllh, obsI, stats, seq, seq_rllh)

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
    model_class.estimate_max_debug = _estimate_max_debug_wrapper
