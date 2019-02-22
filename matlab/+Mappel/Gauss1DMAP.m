% Gauss1DMAP.m
%
% A Mappel point emitter model iterface for:
%  * Model: Gauss1DModel a 1D Gaussian PSF with fixed psf_sigma [sigmaX]
%  * Objective: PoissonNoise1DObjective - Assumes Poisson noise model.
%  * Estimator: MAPEstimator - Maximum a-posteriori likelihood function, that incorporates prior information.
%
% Notes: 
%  * These estimators are designed to work on Poisson distributed data.
%      * All image data should be calibrated to ensure the Poisson noise assumption holds [at least approximately].
%
% Methods and Properties:
% See also Mappel.MappelBase

% Mark J. Olah (mjo@cs.unm DOT edu)
% 2014 - 2019
% COPYRIGHT: See: LICENCE

classdef Gauss1DMAP < Mappel.MappelBase
    properties (Constant=true)
        Name = 'Gauss1DMAP';
        ImageDim = 1;
    end% public constant properties

    properties (Access=public, Constant=true)
        DefaultParamUnits={'pixels','#','#'};
        DefaultParamDescription={'x-position', 'Intensity', 'background'};
    end % public constant properties

    methods (Access=public)
        function obj = Gauss1DMAP(imsize, psf_sigma)
            % obj = Gauss1DMAP(imsize,psf_sigma) - Make a new Gauss1DMAP for
            % point localization in 2D with a fixes PSF.
            % (in) imsize: scalar int - size of image in pixels
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object      
            obj@Mappel.MappelBase(@Gauss1DMAP_IFace, imsize, psf_sigma);
            % set defaults
            obj.ParamUnits = obj.DefaultParamUnits;
            obj.ParamDescription = obj.DefaultParamDescription;
            obj.GPUGaussMLEFitType = -1; %Disabled
        end
    end %public methods
end % classdef
