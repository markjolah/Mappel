# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel pytest and hypothesis based testing

import numpy
import mappel
import pytest
import hypothesis
import hypothesis.strategies as strategies

def test_size():
    size = 8
    psf = 1
    M = mappel.Gauss1DMLE(8,1)
    assert(size == M.size)
    size=10
    M.size=size
    assert(size == M.size)

