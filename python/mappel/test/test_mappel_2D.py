# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 2D model tests

import pytest

def test_num_dim_2D(model2D):
    """Check model.num_dim for 2D models."""
    assert(model1D.num_dim == 2)
    
def test_size_2D(model2D):
    """check size get and set properties"""
    model = model2D

    assert(model2D.size > 0)
    size = model2D.size
    model1D.size += 1
    assert(size+1 == model1D.size)
    
def test_min_size_1D(model1D):
    """check size error conditions"""
    model = model2D

    assert(model1D.min_size > 0)
    with pytest.raises(ValueError, message="Expecting min_size to be respected"):
        model1D.size = model1D.min_size-1

