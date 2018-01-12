# -*- coding: utf-8 -*-
"""
MappelBase.py
Created on Thu Jan 11 16:13:40 2018

@author: prelich
"""
class Mappel:
     imsize = []
     psf_sigma = []
     
     def __init__(self, imsize=(8,8), psf_sigma=(1,1)):
          self.imsize = imsize
          self.psf_sigma = psf_sigma

     def getStats():
          return
     
     def getHyperParameters():
          return
     
     def setHyperParameters(hyperparams):
          return
     
     def samplePrior(count):
          return
     
     def boundedTheta(theta):
          return
     
     def thetaInBounds(theta):
          return
     
     def modelImage(theta):
          return
     
#     def modelDipImage(theta):
#          # This may be depricated for python
#          return
     
     def simulateImage(theta,count):
          return
     
#     def simulateDipImage():
#          return
     
     def LLH(image, theta):
          return
     
     def modelGrad(image, theta):
          return
     
     def modelHessian(image, theta):
          return
     
     def modelObjective(image, theta, negate):
          return
     
     def modelPositiveHessian(image, theta):
          return
     
     def CRLB(theta):
          return
     
     def estimationAccuracy(theta):
          return
     
     def fisherInformation(theta):
          return
     
     def observedInformation(im, theta):
          return
     
     def scoreFunction(im, theta):
          return
     
     def estimate(image, estimator_name, theta_init):
          return
     
     def estimateDebug(image, estimator_name, theta_init):
          return
     
     def estimatePosterior(image, max_samples, theta_init):
          return
     
     def estimagePosteriorDebug(image, max_samples, theta_init):
          return
     
     def uniformBackgroundModelLLH(ims):
          return
     
     def modelComparisonUniform(alpha, ims, theta_mle):
          return
     
     def noiseBackgroundModelLLH(ims):
          return
     
     def modelComparisonNoise(alpha, ims, theta_mle):
          return
     
     def evaluateEstimatorAt(estimator, theta, nTrials, theta_init):
          return
     
     def evaluateEstimatorOn(estimator, images):
          return
     
     def mapEstimatorAccuracy(estimator, sample_grid):
          return
     
     def makeThetaGridSamples(theta, gridsize, nTrials):
          return
     
     def superResolutionModel(theta, theta_err, res_factor):
          return
     
     ## These are protected methods but Python protects nothing!
     def __estimate_GPUGaussMLE(image):
          return
     
     def __estimateDebug_GPUGaussMLE(image):
          return
     
     def __estimate_fminsearch(image, theta_init):
          return
     
     def __estimateDebug_fminsearch(image, theta_init):
          return
     
     def __estimate_toolbox(image, theta_init, algorithm):
          return
     
     def __estimateDebug_toolbox(image, theta_init, algorithm):
          return
     
     ## these are static methods from the Matlab MappelBase class
     def __cholesky(A):
          return
     
     def __modifiedCholesky(A):
          return
     
     def __choleskySolve(A,b):
          return
     
     def __viewDipImage(image, fig):
          return
     
     def __checkImage(image):
          return
     
     def __checkCount(count):
          return
     
     def __checkTheta(in_theta):
          return
     
     def __checkThetaInit(theta_init, nIms):
          return
     
     def __paramMask(names):
          return
     
     