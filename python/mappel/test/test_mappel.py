# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# Test the properties and non-numerical methods of the model
#

import numpy as np
import pytest

from . common import check_stats

def test_name(model):
    """Check model.name is same as python class name."""
    assert model.name == model.__class__.__name__

def test_num_pixels(model):
    """Check model.num_pixels is correct for model.size"""
    assert model.num_pixels == np.prod(model.size)

def test_esimator_names(model):
    """Check model.estimator_names is a list of strings"""
    assert len(model.estimator_names) > 0

def test_num_params(model):
    """Check model.num_params is non-zero"""
    assert model.num_params > 0

def test_num_hyperparams(model):
    """Check model.num_hyperparams is non-zero"""
    assert model.num_hyperparams > 0

def test_params_desc(model):
    """Check model.params_desc"""
    assert len(model.params_desc) == model.num_params

def test_hyperparams_desc(model):
    """Check model.hyperparams_desc"""
    assert len(model.hyperparams_desc) == model.num_hyperparams

def test_hyperparams(model):
    """Check model.hyperparams property get and set methods"""
    assert len(model.hyperparams) == model.num_hyperparams
    for n in range(model.num_hyperparams):
        params = model.hyperparams
        params[n] /= 1.1
        model.hyperparams = params
        assert all(model.hyperparams == params)

def test_ubound(model):
    """Check model.ubound property get and set methods"""
    assert len(model.ubound) == model.num_params 
    for n in range(model.num_params):
        bd = model.ubound
        bd[n] = 10.
        model.ubound = bd
        assert all(model.ubound == bd)

def test_lbound(model):
    """Check model.lbound property get and set methods"""
    assert len(model.lbound) == model.num_params
    for n in range(model.num_params):
        bd = model.lbound
        bd[n] = 2.
        model.lbound = bd
        assert all(model.lbound == bd)

def test_stats(model):
    """Check model.get_stats() for sanity."""
    stats = model.get_stats()
    check_stats(stats)

