"""
Mappel pytest configuration and fixtures
"""

import numpy as np
import mappel
import pytest

MappelModels1DFixed = [mappel.Gauss1DMLE, mappel.Gauss1DMAP]
MappelModels1D = MappelModels1DFixed
MappelModels = MappelModels1D



MappelConstructorArgs = {
    type(mappel.Gauss1DMLE):{"size":10, "psf_sigma":1.2},
    #type(mappel.Gauss1DMAP):{"size":7, "psf_sigma":0.9},
    #type(mappel.Gauss1DsMLE):{"size":10, "min_sigma":0.9, "max_sigma":3.3},
    #type(mappel.Gauss1DsMAP):{"size":12, "min_sigma":1.2, "max_sigma":3.8}
}

#MappelEstimatorTestMethods = ["heuristic","newton","newtondiagonal","quasinewton","simulatedannealing","trustregion"]
#MappelEstimatorTestMethods = ["heuristic","newton","newtondiagonal","simulatedannealing"]
MappelEstimatorTestMethods = ["heuristic","newton","newtondiagonal"]

def model_id(model_class):
    return model_class.__name__

@pytest.fixture(params=MappelModels1DFixed, ids=model_id)
def model1DFixed(request):
    """Provide an initialized model object. """
    kwargs = MappelConstructorArgs[type(request.param)]
    return request.param(**kwargs)

@pytest.fixture(params=MappelModels1D, ids=model_id)
def model1D(request):
    """Provide an initialized model object. """
    kwargs = MappelConstructorArgs[type(request.param)]
    return request.param(**kwargs)

@pytest.fixture(params=MappelModels, ids=model_id)
def model(request):
    """Provide an initialized model object. """
    kwargs = MappelConstructorArgs[type(request.param)]
    return request.param(**kwargs)
