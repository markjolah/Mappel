classdef Gauss2DsMLE < MappelBase
    properties (Constant=true)
        nParams=5;
        nHyperParams=6;
        Name='Gauss2DsMLE';
        ParamNames={'x', 'y', 'I', 'bg', 'sigma'};
        HyperParamNames= {'Beta_pos', 'Mean_I', 'Kappa_I', 'Mean_bg', 'Kappa_bg','alpha_sigma'};
        ParamUnits={'pixels','pixels','#','#'};
        ParamDescription={'x-position', 'y-position', 'Intensity', 'background', 'apparent sigma'};
    end % constant properties

    properties (Access=protected)
        GPUGaussMLEFitType=2;
    end

    methods (Access=public)
        function obj = Gauss2DsMLE(imsize_,psf_sigma_)
            % obj = Gauss2DsMLE(imsize,psf_sigma) - Make a new Gauss2DsMLE for
            % point localization in 2D with a fixes PSF.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object
            obj@MappelBase(@Gauss2DsMLE_Iface,imsize_,psf_sigma_);
        end
    end %public methods
end % classdef
