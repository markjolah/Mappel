classdef BlinkHSsMAP< MappelHSBase
    properties 
        nParams;
        nHyperParams=11;
        Name='BlinkHSsMAP';
        ParamNames={'x', 'y', 'L', 'I', 'bg', 'sigma', 'sigmaL'};
        HyperParamNames= {'beta_pos', 'beta_L', 'mean_I', 'kappa_I', 'mean_bg', 'kappa_bg',...
                          'alpha_sigma', 'mean_sigmaL', 'xi_sigmaL',  'beta_D0', 'beta_D1'};
        ParamUnits={'pixels','pixels','nm','#','#','pixels','pixels'};
        ParamDescription={'x-position', 'y-position','L-position', 'Intensity', 'background', 'Sigma', 'SigmaL'};
    end % constant properties

    methods (Access=public)
        function obj = BlinkHSsMAP(imsize_,psf_sigma_, lambda_)
            % obj = BlinkHSDsMAP(imsize,psf_sigma) - Make a new BlinkHSsMAPfor
            % point localization in 3D with a assymetrical PSF.
            % (in) imsize: 1x3 [int] - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: 1x3 [double]>0 - size of PSF in pixels
            % (in) lambda: 1x(size(3)+1) [double]>0 - lambdas at each pixel edge from 0
            %       to size(3)+1.  Must be monotonically increasing.
            % (out) obj - A new object
            obj@MappelHSBase(imsize_,psf_sigma_);
            obj.nParams=7+imsize_(1);
            obj.ParamNames=[obj.ParamNames arrayfun(@(i) sprintf('D%i',i), 1:imsize_,'Uniform',0)];
            obj.makeCObj(@BlinkHSsMAP_Iface, int32(obj.imsize), obj.psf_sigma);
        end
    end %public methods
    
%     methods (Access=protected)
%         function theta=checkTheta(obj, theta)
%             if size(theta,1) ~= obj.nParams
%                 error('BlinkHSsMAP:checkTheta', 'Invalid theta shape');
%             end
%             ok =       all( theta(1,:) >= 0. );
%             ok = ok && all( theta(1,:) <= obj.imsize(1) );
%             ok = ok && all( theta(2,:) >  0. );
%             ok = ok && all( theta(2,:) <= obj.imsize(2) );
%             ok = ok && all( theta(3,:) >= obj.lambda(1) );
%             ok = ok && all( theta(3,:) <= obj.lambda(end) );
%             ok = ok && all( theta(4,:) >  0. );
%             ok = ok && all( theta(5,:) >= 0. );
%             ok = ok && all( theta(6,:) >= 1 );
%             ok = ok && all( theta(7,:) > 0. );
%             for i=1:obj.imsize(1)
%                 ok = ok && all( theta(7+i,:) >= 0. ) && all( theta(7+i,:) <= 1. );
%             end
%             if ~ok
%                 error('BlinkHSsMAP:checkTheta', 'Invalid theta');
%             end
%             theta=double(theta);
%         end
%     end % protected methods
end % classdef
