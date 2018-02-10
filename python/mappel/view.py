# mappel/view.py
#
# Peter Relich (prelich\@upenn DOT edu)
#
# Viewing methods for Mappel Models
import numpy as np
from matplotlib import pyplot as plt
import mappel
import matplotlib as mpl
from scipy.optimize import least_squares
from scipy.special import erf

class viewer1D:
    def __init__(self,imsize=8,psf_sigma=1.0):
        self.imsize = imsize
        self.psf_sigma = psf_sigma
        self.engine = mappel.Gauss1DMLE(imsize,psf_sigma)

    def genModelSimData(self,thetas=None):
        if thetas==None:
            thetas = P_samp = self.engine.sample_prior(1)
        self.thetas = thetas
        self.sim = self.engine.simulate_image(thetas)
        self.model = self.engine.model_image(thetas)

    def overlayModelSim(self,thetas=None):
        self.genModelSimData(thetas)
        centroid = self.calcCentroid(self.sim)
        lsqE = self.leastSquares(self.sim)
        #P_samp = self.engine.sample_prior(1)
        I1 = self.sim.sum()
        b1 = self.sim.min()
        [MLE,LLH,Hess] = self.engine.estimate_max(self.sim,'Newton',[centroid,I1,b1])
        psf_sig = self.engine.psf_sigma
        Normerf = 0.5*(erf( (np.ceil(self.thetas[0])-self.thetas[0])/np.sqrt(2)/psf_sig ) - erf( (np.floor(self.thetas[0])-self.thetas[0])/np.sqrt(2)/psf_sig ) )
        # perform plotting of histograms and estimators
        fig, ax = plt.subplots()
        fig_fig_im = ax.bar(range(self.sim.size),self.sim,color='r',label='simulated')
        fig_md = ax.bar(range(self.model.size),self.model,color='b',alpha=0.4,label='model')
        fig_centroid = ax.plot(centroid-0.5,self.sim.max(),'k+',markersize=15,label='Centroid')
        fig_lsq = ax.plot(lsqE[0]-0.5,lsqE[1]*Normerf+lsqE[2],'rx',markersize=15,label='Least Squares')
        fig_mle = ax.plot(MLE[0]-0.5,MLE[1]*Normerf+MLE[2],'bo',markersize=15,label='Maximum Likelihood')
        ax.set_title('Simulated and Model Histograms with Point Estimates')
        ax.set_xlabel('pixel position')
        ax.set_ylabel('pixel count')
        ax.legend()
        plt.show(fig)

    def calcCentroid(self,data):
        xpos = 0
        for count, elem in enumerate(data):
            xpos = xpos+count*elem
        return xpos/data.sum()

    def leastSquares(self,data):
        x_init = self.calcCentroid(data)
        I_init = data.sum()
        bg_init = data.min()
        thetas = [x_init, I_init, bg_init]
        def loss_Func(thetas):
            model = self.engine.model_image(thetas)
            return np.sum( (data-model)**2)
        res_1 = least_squares(loss_Func,thetas,bounds=([0,0,0],[self.imsize,4*I_init,4*I_init]))
        return res_1.x
    
class viewer2D:
    def __init__(self,imsize=[8,8],psf_sigma=[1.0,1.0]):
        self.imsize = imsize
        self.psf_sigma = psf_sigma
        self.engine = mappel.Gauss2DMLE(imsize,psf_sigma)

    def _view2D(self,data):
        pg.image(data,title="2 Dimensional Data")
        return
