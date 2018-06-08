# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# These tests use the hypothesis strategies.data() type to
# generate strictly feasible parameter values which test out
# very large and very small values.

import numpy as np
import scipy.linalg
import pytest
import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from hypothesis import settings

from .common import *

settings.register_profile("feasible_samples", max_examples=50)
settings.load_profile("feasible_samples")

@hypothesis.given(data=st.data())
def test_model_image(model, data):
    """Check the model_image returns sane images"""
    theta = draw_feasible_theta(model,data)
    im = model.model_image(theta)
    check_image(model,im,1)

#@hypothesis.given(data=st.data())
#def test_simulated_image(model, data):
    #"""Check the model_image returns sane images"""
    #theta = draw_feasible_theta(model,data)
    #im = model.simulate_image(theta)
    #check_image(model,im)

@hypothesis.given(data=st.data())
def test_prior_objective(model, data):
    """Check model.prior_objective() behaves within the bounds"""
    theta = draw_feasible_theta(model,data)
    (rllh,grad,hess) = model.prior_objective(theta);
    assert math.isfinite(rllh)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
    
@hypothesis.given(data=st.data())
def test_theta_in_bounds_feasible(model, data):
    """Check model.theta_in_bounds works for feasible theta"""
    theta = draw_feasible_theta(model,data)
    assert np.all(model.theta_in_bounds(theta))

#@hypothesis.given(data=st.data())
#def test_theta_in_bounds_infeasible(model, data):
    #"""Check model.theta_in_bounds works for infeasible theta"""
    #theta = draw_infeasible_theta(model,data)
    #assert not np.all(model.theta_in_bounds(theta))
    
#@hypothesis.given(data=st.data())
#def test_reflected_theta_feasible(model, data):
    #"""Check model.reflected_theta works for feasible"""
    #theta = draw_feasible_theta(model,data)
    #rtheta = model.reflected_theta(theta)
    #assert np.all(theta == rtheta)

#@hypothesis.given(data=st.data())
#def test_reflected_theta_infeasible(model, data):
    #"""Check model.reflected_theta for  infeasible"""
    #theta = draw_infeasible_theta(model,data)
    #rtheta = model.reflected_theta(theta)
    #assert np.all(model.in_bounds(rtheta))
    
#@hypothesis.given(data=st.data())
#def test_bounded_theta_feasible(model, data):
    #"""Check model.bounded_theta works for feasible"""
    #theta = draw_feasible_theta(model,data)
    #rtheta = model.bounded_theta(theta)
    #assert np.all(theta == rtheta)

#@hypothesis.given(data=st.data())
#def test_bounded_theta_infeasible(model, data):
    #"""Check model.reflected_theta for  infeasible"""
    #theta = draw_infeasible_theta(model,data)
    #rtheta = model.reflected_theta(theta)
    #assert np.all(model.in_bounds(rtheta))
    


