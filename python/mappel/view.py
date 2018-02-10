# mappel/view.py
#
# Peter Relich (prelich\@upenn DOT edu)
#
# Viewing methods for Mappel Models
import numpy as np
from matplotlib import pyplot as plt
import mappel
import matplotlib as mpl

class viewer1D(imsize=8,psf_sigma=1.0):
    def __init__(self):
        self.imsize = imsize
        self.psf_sigma = psf_sigma
        self.engine = mappel.Gauss1DMLE(imsize,psf_sigma)

    def genModelSimData(self,thetas=None):
        if thetas==None:
            thetas = P_samp = M.sample_prior(1)
        self.sim = self.engine.simulate_image(thetas)
        self.model = self.engine.model_image(thetas)

    def overlayModelSim(self,thetas=None):
        self.genModelSimData(thetas)
        

    def calcCentroid(data):
        xpos = 0
        for count, elem in enumerate(data):
            xpos = xpos+count*elem/elem
        return xpos

    def leastSquares(data):

        return

    
class viewer2D(imsize=[8,8],psf_sigma=[1.0,1.0])
    def __init__(self):
        self.imsize = imsize
        self.psf_sigma = psf_sigma
        self.engine = mappel.Gauss2DMLE(imsize,psf_sigma)

    def _view2D(self,data):
        pg.image(data,title="2 Dimensional Data")
        return
