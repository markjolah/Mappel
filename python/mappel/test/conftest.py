"""
Mappel pytest configuration and fixtures
"""
import numpy as np
import mappel
import pytest

#Models1DFixedSigma = [mappel.Gauss1DMLE, mappel.Gauss1DMAP]
Models1DFixedSigma = []
#Models1DVariableSigma = [mappel.Gauss1DsMLE, mappel.Gauss1DsMAP]
Models1DVariableSigma = []
Models2DFixedSigma = [mappel.Gauss2DMLE]
Models2DVariableSigma = []

Models1D = Models1DFixedSigma + Models1DVariableSigma
Models2D = Models2DFixedSigma + Models2DVariableSigma
Models = Models1D + Models2D



ModelConstructorArgs = {
    mappel.Gauss1DMLE:{"size":10, "psf_sigma":1.2},
    mappel.Gauss1DMAP:{"size":7, "psf_sigma":0.9},
    mappel.Gauss1DsMLE:{"size":10, "min_sigma":0.9, "max_sigma":3.3},
    mappel.Gauss1DsMAP:{"size":12, "min_sigma":1.2, "max_sigma":3.8},
    mappel.Gauss2DMLE:{"size":np.array([8,10],dtype="uint32"), "psf_sigma":np.array([0.9,1.2],dtype="double")},
}

MappelEstimatorTestMethods = ["heuristic","newton","newtondiagonal","quasinewton","simulatedannealing","trustregion"]
#MappelEstimatorTestMethods = ["heuristic","newton","newtondiagonal","simulatedannealing"]

def model_id(model_class):
    return model_class.__name__

@pytest.fixture(params=Models1DFixedSigma, ids=model_id)
def model1DFixedSigma(request):
    """Provide an initialized 1D fixed sigma model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models1DVariableSigma, ids=model_id)
def model1DVariableSigma(request):
    """Provide an initialized 1D variable sigma model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models1D, ids=model_id)
def model1D(request):
    """Provide an initialized 1D model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models2DFixedSigma, ids=model_id)
def model2DFixedSigma(request):
    """Provide an initialized 2D fixed sigma model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models2DVariableSigma, ids=model_id)
def model2DVariableSigma(request):
    """Provide an initialized 2D variable sigma model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models2D, ids=model_id)
def model2D(request):
    """Provide an initialized 2D model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)

@pytest.fixture(params=Models, ids=model_id)
def model(request):
    """Provide an initialized model object. """
    kwargs = ModelConstructorArgs[request.param]
    return request.param(**kwargs)
