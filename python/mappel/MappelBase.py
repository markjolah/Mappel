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

    def getStats(self):
        return self.get_stats
    
    # Return a dictionary of keys hyperparam_desc and values hyperparams
     def getHyperParameters(self):
         keys = self.hyperparams_desc
         values = self.hyperparams
         return dict(zip(keys, values))
    
#     def setHyperParameters(hyperparams):
#        return
    
    def samplePrior(count=1):
        return self.sample_prior(count)

    def boundedTheta(theta):
        return self.bounded_theta(theta)

    def thetaInBounds(theta):
        return self.theta_in_bounds(theta)

    def modelImage(theta):
        return self.modelImage(theta)
    
    def simulateImage(self,count=1,theta=None):
        if theta is None:
            theta = self.sample_prior(count)
        return self.simulate_image(theta)
    
    def LLH(image, theta):
        self.__checkinputs({"image":image,"theta":theta})
        return self.objective_llh(image,theta)
    
    def modelGrad(image, theta):
        self.__checkinputs({"image":image, "theta":theta})
        return self.objective_grad(image,theta)
    
    def modelHessian(image, theta):
        self.__checkinputs({"image":image, "theta":theta})
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
        self.__checkinputs({"image":image, "theta":theta})
        return self.observed_information(image, theta)
    
#     def scoreFunction(im, theta):
#        return
    
    def estimate(self, image, theta_init, estimator_name='Newton'):
        self.__checkinputs({"image":image, "theta_init":theta_init})
        return self.estimate_max(image, estimator_name, theta_init)
    
    def estimateDebug(image, theta_init, estimator_name="Newton"):
        self.__checkinputs({"image":image, "theta_init":theta_init})
        return self.estimate_max_debug(image, estimator
    
#     def estimatePosterior(image, max_samples, theta_init):
#        return
    
#     def estimagePosteriorDebug(image, max_samples, theta_init):
#        return
    
#     def superResolutionModel(theta, theta_err, res_factor):
#        return
    
    ## These are protected methods but Python protects nothing!
    
    # all input checks get routed through this function
    def __check_inputs(input_dict):
        for key, value in input_dict:
            print(key)
        return

#     def __estimate_GPUGaussMLE(image):
#        return
    
#     def __estimateDebug_GPUGaussMLE(image):
#        return
    
    ## these are static methods from the Matlab MappelBase class
    
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
    
    
