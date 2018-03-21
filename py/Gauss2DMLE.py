# -*- coding: utf-8 -*-
"""
Gauss2DMLE.py
Created on Thu Jan 11 15:47:17 2018

@author: prelich
"""
from MappelBase import Mappel

## begin class declaration here
class Gauss2DMLE(Mappel):

     'Documentation pending'
     Name = 'Gauss2DMLE'
     nParams = 4
     ParamNames = ('x','y','I','bg')
     ParamUnits = ('pixels','pixels','counts','counts')
     ParamDescription = ('Beta_pos', 'Mean_I', 'Kappa_I', 'Mean_bg', 'Kappa_bg')
     
     def __init__(self, imsize=(8,8), psf_sigma=(1,1)):
          Mappel.__init__(self) #, imsize, psf_sigma)
     