classdef Blink2DsMAP< MappelBase
    properties 
        Name='Blink2DsMAP';
        nParams;
        ParamNames={'x', 'y', 'I', 'bg', 'sigma'};
        ParamUnits={'pixels','pixels','#','#','pixels'};
        ParamDescription={'x-position', 'y-position', 'Intensity', 'background','Apparent PSF Sigma'};
        nHyperParams=8;
        HyperParamNames= {'beta_pos', 'mean_I', 'kappa_I', 'mean_bg', 'kappa_bg', 'alpha_sigma', 'beta_D0', 'beta_D1'};
    end % constant properties

    properties (Access=protected)
        GPUGaussMLEFitType=2;
    end

    methods (Access=public)
        function obj = Blink2DsMAP(imsize_,psf_sigma_)
            % obj = Blink2DsMAP(imsize,psf_sigma) - Make a new Blink2DsMAPfor
            % point localization in 2D with a fixes PSF.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object
            import CellFun.*
            obj@MappelBase(@Blink2DsMAP_Iface,imsize_,psf_sigma_);
            obj.nParams=5+imsize_(1);
            obj.ParamNames=[obj.ParamNames arrayfun(@(i) sprintf('D%i',i), 1:imsize_, 'Uniform', 0)];
            obj.ParamDescription=[obj.ParamDescription arrayfun(@(i) sprintf('Duty Ratio Col:%i',i), 1:imsize_, 'Uniform', 0)];
            obj.ParamUnits=[obj.ParamUnits arrayfun(@(i) '-', 1:imsize_, 'Uniform', 0)];
        end
    end %public methods
    
end % classdef
