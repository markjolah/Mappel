% Gauss2DsMAP.m
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

classdef Gauss2DsMAP < Mappel.MappelBase
    properties (Constant=true)
        Name = 'Gauss2DsMAP';
        ImageDim = 2;
    end% public constant properties

    properties (Access=public, Constant=true)
        DefaultParamUnits={'pixels','pixels','#','#','ratio'};
        DefaultParamDescription={'x-position', 'y-position', 'Intensity', 'background','sigma-ratio'};
    end % public constant properties

    properties (Access=protected, Constant=true)
        DefaultGPUGaussMLEFitType=2; %Fitting mode used for gpugaussmle estimator comparison
    end % protected constant properties

    properties (Access=public)
        GPUGaussMLE_Iterations
    end
    
    methods (Access=public)
        function obj = Gauss2DsMAP(imsize, psf_min, psf_max)
            % obj = Gauss2DsMAP(imsize,psf_min,psf_max) - Make a new Gauss2DsMAP for
            % point localization in 2D with a linearly variable psf.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_min: scalar or size:[2].  minimum psf_sigma.  Can be non-isometric
            %                psf_min(1)~=psf_min(2).
            % (in) psf_max: scalar or size:[2].  If scalar must be >1 and represents max ratio of psf_min.
            %               if size:[2], then must be an exact (>1) multiple of psf_min.
            % (out) obj - A new object      
            obj@Mappel.MappelBase(@Gauss2DsMAP_IFace, imsize, psf_min, psf_max);
            % set defaults
            obj.ParamUnits = obj.DefaultParamUnits;
            obj.ParamDescription = obj.DefaultParamDescription;
            obj.GPUGaussMLEFitType = obj.DefaultGPUGaussMLEFitType;
            obj.GPUGaussMLE_Iterations = obj.DefaultGPUGaussMLE_Iterations;
        end
    end %public methods
end % classdef

