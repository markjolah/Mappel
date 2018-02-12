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
    
    def genMLEstatistics(self,data):
        centroid = self.calcCentroid(self.sim)
        I1 = self.sim.sum()
        b1 = self.sim.min()
        return self.engine.estimate_max(self.sim,'Newton',[centroid,I1,b1])

    def overlayHessianLikelihood(self,thetas=None):
        self.genModelSimData(thetas)
        [mle_vals, llh_max, Hess] = self.genMLEstatistics(self.sim)
        # generate log likelihood curve
        xs = np.arange(mle_vals[0]-1,mle_vals[0]+1,0.01)
        # fix I and bg at MLE value at MLE of x for now
        ism = mle_vals[1]*np.ones(xs.size)
        bsm = mle_vals[2]*np.ones(xs.size)
        thetas = (xs,ism,bsm)
        llh = self.engine.objective_llh(self.sim,thetas)
        max_llh = self.engine.objective_llh(self.sim,mle_vals)
        # generate hessian curve
        hsc = -Hess[0,0]*(xs-mle_vals[0])**2+max_llh
        # plot the curves
        pparams = self.plotParams()
        font = pparams['font']
        mpl.rc('font', **font)
        fig,ax = plt.subplots()
        fig_llh = ax.plot(xs,llh,'r+',ms='8',mew='2',linewidth='5.0',label='log likelihood')
        fig_hess = ax.plot(xs,hsc,'cv',ms='8',mew='2',linewidth='5.0',label='information parabola')

        ax.set_title('Log Likelihood vs Information Parametric')
        ax.set_xlabel('pixel position')
        ax.set_ylabel('log likelihood')
        ax.legend()

        sim_string = "truth: position={0:.2f}, intensity={1:.2f}, background={2:.2f}".format(self.thetas[0],self.thetas[1],self.thetas[2])
        ax.text(0.95, 0.1, sim_string, verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=20, bbox={'facecolor':'cyan', 'alpha':0.5, 'pad':10})

        plt.show(fig)

    def compareLikelihoodtoData(self,thetas=None):
        self.genModelSimData(thetas)
        [mle_vals, llh, Hess] = self.genMLEstatistics(self.sim)
        # get sampling area
        xs = np.arange(0.01,8,0.01)
        # fix I and bg at MLE value at MLE of x for now
        ism = mle_vals[1]*np.ones(xs.size)
        bsm = mle_vals[2]*np.ones(xs.size)
        thetas = (xs,ism,bsm)
        # call objective_llh
        llh = self.engine.objective_llh(self.sim,thetas)
        # perform plotting
        pparams = self.plotParams()
        font = pparams['font']
        mpl.rc('font', **font)
        fig,(ax0,ax1) = plt.subplots(1,2)
        fig_im = ax0.bar(range(self.sim.size),self.sim,color='r',label='simulated')
        fig_llh = ax1.plot(xs,llh,linewidth=5.0)

        ax0.set_title('Simulated Histogram')
        ax0.set_xlabel('pixel position')
        ax0.set_ylabel('pixel count')
        
        ax1.set_title('Log Likelihood of X given the Histogram')
        ax1.set_ylabel('Log Likelihood')
        ax1.set_xlabel('X position of PSF Model')

        plt.show(fig)

    def overlayLeastSquares(self,thetas=None):
        self.genModelSimData(thetas)
        centroid = self.calcCentroid(self.sim)
        lsqE = self.leastSquares(self.sim)

        sampling_freq = 100
        gauss_simulated = np.zeros(self.sim.size*sampling_freq)
        gauss_model = np.zeros(self.sim.size*sampling_freq)
        normfunc = lambda x,x0,I0,bg0,sig: I0/np.sqrt(2*np.pi) * np.exp(-(x-x0)**2/2/sig**2) + bg0
        for ii in range(gauss_simulated.size):
             gauss_simulated[ii] = normfunc(ii/sampling_freq,lsqE[0],lsqE[1],lsqE[2],self.engine.psf_sigma)
             gauss_model[ii] = normfunc(ii/sampling_freq,lsqE[0],lsqE[1],lsqE[2],self.engine.psf_sigma)
        # perform plotting
        pparams = self.plotParams()
        font = pparams['font']
        mpl.rc('font', **font)
        fig, ax = plt.subplots()
        fig_im = ax.bar(range(self.sim.size),self.sim,color='r',label='simulated')
        fig_imgauss = ax.plot(np.arange(gauss_simulated.size)/sampling_freq-0.5,gauss_simulated,label='Gaussian Fit',linewidth=5.0)
        ax.set_title('Least Squares Fit over Simulated Histogram')
        ax.set_xlabel('pixel position')
        ax.set_ylabel('pixel count')
        ax.legend()
        plt.show(fig)

    def overlayModelSim(self,thetas=None):
        self.genModelSimData(thetas)
        centroid = self.calcCentroid(self.sim)
        lsqE = self.leastSquares(self.sim)
        [MLE,LLH,Hess] = self.genMLEstatistics(self.sim)
        psf_sig = self.engine.psf_sigma
        Normerf = 0.5*(erf( (np.ceil(self.thetas[0])-self.thetas[0])/np.sqrt(2)/psf_sig ) - erf( (np.ceil(self.thetas[0])-self.thetas[0]-1)/np.sqrt(2)/psf_sig ) )
        # perform plotting of histograms and estimators
        pparams = self.plotParams()
        font = pparams['font']
        mpl.rc('font', **font)
        
        fig, ax = plt.subplots()
        fig_fig_im = ax.bar(np.arange(0.5,self.sim.size,1),self.sim,color='r',label='simulated')
        fig_md = ax.bar(np.arange(0.5,self.model.size,1),self.model,color='b',alpha=0.4,label='model')
        fig_truth = ax.plot(self.thetas[0],self.thetas[1]*Normerf+self.thetas[2],'k*',ms=25,mew=6,label='Simulation Truth')
        fig_centroid = ax.plot(centroid+0.5,self.sim.max(),'c+',ms=25,mew=6,label='Centroid')
        fig_lsq = ax.plot(lsqE[0],lsqE[1]*Normerf+lsqE[2],'mx',ms=25,mew=6,label='Least Squares')
        fig_mle = ax.plot(MLE[0],MLE[1]*Normerf+MLE[2],'g.',ms=25,mew=6,label='Maximum Likelihood')
        ax.set_title('Simulated and Model Histograms with Point Estimates')
        ax.set_xlabel('pixel position')
        ax.set_ylabel('pixel count')
        ax.legend()
        
        sim_string = "truth: position={0:.2f}, intensity={1:.2f}, background={2:.2f}".format(self.thetas[0],self.thetas[1],self.thetas[2])
        ax.text(0.95, 0.1, sim_string, verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=20, bbox={'facecolor':'cyan', 'alpha':0.5, 'pad':10}) 
       
        plt.show(fig)

    def plotParams(self):
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size' : 22}
        return {'font':font}

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
