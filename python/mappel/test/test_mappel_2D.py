# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 2D model tests

import pytest
import numpy as np

def test_num_dim_2D(model2D):
    """Check model.num_dim for 2D models."""
    assert model2D.num_dim == 2
    
def test_size_2D(model2D):
    """check size get and set properties"""
    model = model2D
    assert np.all(model.size >= model.global_min_size)
    for n in range(model.num_dim):
        size = model.size;
        size[n] += 1
        model.size = size
        assert np.all(size == model.size)
    
def test_min_size_2D(model2D):
    """check size error conditions"""
    model = model2D
    assert model.global_min_size > 0
    for n in range(model.num_dim):
        with pytest.raises(ValueError, message="Expecting min_size to be respected"):
            size = model.size 
            size[n] = model.global_min_size-1
            model.size = size

