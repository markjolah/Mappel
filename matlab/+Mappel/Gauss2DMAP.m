% Gauss2DMAP.m
% Mark J. Olah (mjo@cs.unm DOT edu)
% 2014 - 2019
% COPYRIGHT: See: LICENCE
%
% A Mappel point emitter model interface for:
%  * Model: Gauss2DModel a 2D Gaussian PSF with fixed psf_sigma [sigmaX, sigmaY]
%  * Objective: PoissonNoise2DObjective - Assumes Poisson noise model.
%  * Estimator: MAPEstimator - Maximum a-posteriori likelihood function, that incorporates prior information.
%
% Notes: 
%  * These estimators are designed to work on Poisson distributed data.
%      * All image data should be calibrated to ensure the Poisson noise assumption holds [at least approximately].
%
% Methods and Properties:
% See: Mappel.MappelBase

classdef Gauss2DMAP < Mappel.MappelBase
    properties (Constant=true)
        Name = 'Gauss2DMAP';
        ImageDim = 2;
    end% public constant properties

    properties (Access=public, Constant=true)
        DefaultParamUnits={'pixels','pixels','#','#'};
        DefaultParamDescription={'x-position', 'y-position', 'Intensity', 'background'};
    end % public constant properties

    properties (Access=protected, Constant=true)
        DefaultGPUGaussMLEFitType=1; %Fitting mode used for gpugaussmle estimator comparison
    end % protected constant properties

    properties (Access=public)
        GPUGaussMLE_Iterations
    end
    
    methods (Access=public)
        function obj = Gauss2DMAP(imsize, psf_sigma)
            % obj = Gauss2DMAP(imsize,psf_sigma) - Make a new Gauss2DMAP for
            % point localization in 2D with a fixed PSF.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object      
            obj@Mappel.MappelBase(@Gauss2DMAP_IFace, imsize, psf_sigma);
            % set defaults
            obj.ParamUnits = obj.DefaultParamUnits;
            obj.ParamDescription = obj.DefaultParamDescription;
            obj.GPUGaussMLEFitType = obj.DefaultGPUGaussMLEFitType;
            obj.GPUGaussMLE_Iterations = obj.DefaultGPUGaussMLE_Iterations;
        end
    end %public methods
end % classdef
