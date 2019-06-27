% Gauss1DsMAP.m
%
% A Mappel point emitter model iterface for:
%  * Model: Gauss1DsModel a 1D Gaussian PSF with variable psf_sigma measured in pixels
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

classdef Gauss1DsMAP < Mappel.MappelBase
    properties (Constant=true)
        Name = 'Gauss1DsMAP';
        ImageDim = 1;
    end% public constant properties

    properties (Access=public, Constant=true)
        DefaultParamUnits={'pixels','#','#','pixels'};
        DefaultParamDescription={'x-position', 'Intensity', 'background', 'sigma'};
    end % public constant properties

    methods (Access=public)
        function obj = Gauss1DsMAP(imsize, psf_min, psf_max)
            % obj = Gauss1DsMAP(imsize,psf_sigma) - Make a new Gauss1DsMAP for
            % point localization in 2D with a fixes PSF.
            % (in) imsize: scalar int - size of image in pixels
            % (in) psf_min: scalar double - minimum size of psf sigma in pixels
            % (in) psf_max: scalar double - maximum size of psf sigma in pixels
            % (out) obj - A new object      
            obj@Mappel.MappelBase(@Gauss1DsMAP_IFace, imsize, psf_min, psf_max);
            % set defaults
            obj.ParamUnits = obj.DefaultParamUnits;
            obj.ParamDescription = obj.DefaultParamDescription;
            obj.GPUGaussMLEFitType = -1; %Disabled
        end
    end %public methods
end % classdef
