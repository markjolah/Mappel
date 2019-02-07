% Gauss2DMAP.m
% Mark J. Olah (mjo@cs.unm DOT edu)
% 2014 - 2019
% COPYRIGHT: See: LICENCE
%
% A Mappel point emitter model iterface for:
%  * Model: Gauss2DModel a 2D Gaussian PSF with fixed psf_sigma [sigmaX, sigmaY]
%  * Objective: PoissonNoise2DObjective - Assumes Poisson noise model.
%  * Estimator: MAPEstimator - Maximum a-posteriori likelihood function, that incorporates prior information.
%
% Notes: Data should be calibrated to ensure the Poisson noise assumption holds, at least approximately.
%

classdef Gauss2DMAP < Mappel.MappelBase
    properties (Constant=true)
        Name = 'Gauss2DMAP';
    end% public constant properties

    properties (Access=private, Constant=true)
        DefaultParamUnits={'pixels','pixels','#','#'};
        DefaultParamDescription={'x-position', 'y-position', 'Intensity', 'background'};
        DefaultGPUGaussMLEFitType=1; %Fitting mode used for gpugaussmle estimator comparison
    end % private constant properties
    
    methods (Access=public)
        function obj = Gauss2DMAP(imsize_,psf_sigma_)
            % obj = Gauss2DMAP(imsize,psf_sigma) - Make a new Gauss2DMAP for
            % point localization in 2D with a fixes PSF.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object
            if isscalar(imsize)
                imsize = [imsize, imsize];
            end
            if isscalar(psf_sigma)
                psf_sigma = [psf_sigma, psf_sigma];
            end
            obj@Mappel.MappelBase(@Mappel.Gauss2DMAP_IFace, imsize_, psf_sigma_);
            % set defaults
            obj.ParamUnits = obj.DefaultParamUnits;
            obj.ParamDescription = obj.DefaultParamDescription;
            obj.DefaultGPUGaussMLEFitType = obj.DefaultGPUGaussMLEFitTypel;
        end
    end %public methods
end % classdef
