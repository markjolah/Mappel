# -*- coding: utf-8 -*-
"""
MappelBase.py
Created on Thu Jan 11 16:13:40 2018

@author: prelich
"""
import numpy as np
#import mappel
from _Gauss1DMLE import Gauss1DMLE

class MappelBase(Gauss1DMLE):
    
    # note to self, look into python over-loaded functions
    def __init__(self, imsize, psf_sigma):
        Gauss1DMLE.__init__(self, imsize, psf_sigma)
        self.imsize = imsize
        self.psf_sigma = psf_sigma

#    Pybind11 is near perfect here (self.get_stats)
#     def getStats(self):
#        return
    
    # Return a dictionary of keys hyperparam_desc and values hyperparams
#     def getHyperParameters(self):
#        return
    
#     def setHyperParameters(hyperparams):
#        return
    
    def samplePrior(count=1):
        #if count < 1 or not is instance(count, int):
            #print('Count needs to be an integer of 1 or greater.')
            #pass
        return self.sample_prior(count)

#    pybind11 => self.bounded_theta(theta), no errors triggered yet
#     def boundedTheta(theta):
#        return

#   pybind11 => self.theta_in_bounds(theta)
#     def thetaInBounds(theta):
#        return

#   pybind11 => self.modelImage(theta)
#     def modelImage(theta):
#        return
    
#     def modelDipImage(theta):
#          # This may be depricated for python
#          return
    
    def simulateImage(self,count=1,theta=None):
        if theta is None:
            theta = self.sample_prior(count)
        return self.simulate_image(theta)
    
#     def simulatueDipImage():
#          return
    
    def LLH(image, theta=None):
        if theta is None:
            print('function requires two arguments, image and parameter values')
            pass
        return self.objective_llh(image,theta)
    
    def modelGrad(image, theta):
        # make general function for input checks.
        return self.objective_grad(image,theta)
    
    def modelHessian(image, theta):
        # make general function for input checks
        return self.objective_hessian(image,theta)
    
#     def modelObjective(image, theta, negate):
#        return
    
#     def modelPositiveHessian(image, theta):
#        return
    
#     def CRLB(theta):
#        return
    
#     def estimationAccuracy(theta):
#        return
    
#     def fisherInformation(theta):
#        return
    
    def observedInformation(image, theta):
        # make general function for input checks
        return self.observed_information(image, theta)
    
#     def scoreFunction(im, theta):
#        return
    
    def estimate(self, image, estimator_name='Newton', theta_init=None):
        if theta_init is None:
            theta_init = np.ones([3,image.shape[1]])
        return self.estimate_max(image, estimator_name, theta_init)
    
#     def estimateDebug(image, estimator_name, theta_init):
#        return
    
#     def estimatePosterior(image, max_samples, theta_init):
#        return
    
#     def estimagePosteriorDebug(image, max_samples, theta_init):
#        return
    
#     def uniformBackgroundModelLLH(ims):
#        return
    
#     def modelComparisonUniform(alpha, ims, theta_mle):
#        return
    
#     def noiseBackgroundModelLLH(ims):
#        return
    
#     def modelComparisonNoise(alpha, ims, theta_mle):
#        return
    
#     def evaluateEstimatorAt(estimator, theta, nTrials, theta_init):
#        return
    
#     def evaluateEstimatorOn(estimator, images):
#        return
    
#     def mapEstimatorAccuracy(estimator, sample_grid):
#        return
    
#     def makeThetaGridSamples(theta, gridsize, nTrials):
#        return
    
#     def superResolutionModel(theta, theta_err, res_factor):
#        return
    
    ## These are protected methods but Python protects nothing!
#     def __estimate_GPUGaussMLE(image):
#        return
    
#     def __estimateDebug_GPUGaussMLE(image):
#        return
    
#     def __estimate_fminsearch(image, theta_init):
#        return
    
#     def __estimateDebug_fminsearch(image, theta_init):
#        return
    
#     def __estimate_toolbox(image, theta_init, algorithm):
#        return
    
#     def __estimateDebug_toolbox(image, theta_init, algorithm):
#        return
    
    ## these are static methods from the Matlab MappelBase class
#     def __cholesky(A):
#        return
    
#     def __modifiedCholesky(A):
#        return
    
#     def __choleskySolve(A,b):
#        return
    
#     def __viewDipImage(image, fig):
#        return
    
#     def __checkImage(image):
#        return
    
#     def __checkCount(count):
#        return
    
#     def __checkTheta(in_theta):
#        return
    
#     def __checkThetaInit(theta_init, nIms):
#        return
    
#     def __paramMask(names):
#        return
    
    
