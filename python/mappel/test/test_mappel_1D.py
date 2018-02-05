# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 1D model tests

import pytest

def test_num_dim_1D(model1D):
    """Check model.num_dim for 1D models."""
    assert(model1D.num_dim == 1)
    
def test_size_1D(model1D):
    """check size get and set properties"""
    assert(model1D.size > 0)
    size = model1D.size
    model1D.size += 1
    assert(size+1 == model1D.size)
    
def test_min_size_1D(model1D):
    """check size error conditions"""
    assert(model1D.min_size > 0)
    with pytest.raises(ValueError, message="Expecting min_size to be respected"):
        model1D.size = model1D.min_size-1

