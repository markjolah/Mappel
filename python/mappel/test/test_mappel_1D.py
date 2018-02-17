# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 1D model tests

import pytest

def test_num_dim_1D(model1D):
    """Check model.num_dim for 1D models."""
    assert model1D.num_dim == 1
    
def test_size_1D(model1D):
    """check size get and set properties"""
    model = model1D
    assert model.size >= model.min_size
    size = model.size
    model.size += 1
    assert size+1 == model.size
    
def test_min_size_1D(model1D):
    """check size error conditions"""
    model = model1D
    assert model.min_size > 0
    with pytest.raises(ValueError, message="Expecting min_size to be respected"):
        model.size = model.min_size-1

