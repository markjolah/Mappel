classdef GaussHSsMAP< MappelHSBase
    properties 
        nParams=7;
        nHyperParams=9;
        Name='GaussHSsMAP';
        ParamNames={'x', 'y', 'L', 'I', 'bg', 'sigma', 'sigmaL'};
        HyperParamNames= {'beta_pos', 'beta_L', 'mean_I', 'kappa_I', 'mean_bg', 'kappa_bg',...
                          'alpha_sigma', 'mean_sigmaL', 'xi_sigmaL'};
        ParamUnits={'pixels','pixels','nm','#','#','pixels','pixels'};
        ParamDescription={'x-position', 'y-position','L-position', 'Intensity', 'background', 'Sigma', 'SigmaL'};
    end % constant properties


    methods (Access=public)
        function obj = GaussHSsMAP(imsize_,psf_sigma_)
            % obj = BlinkHSDsMAP(imsize,psf_sigma) - Make a new BlinkHSsMAPfor
            % point localization in 3D with a assymetrical PSF.
            % (in) imsize: 1x3 [int] - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: 1x3 [double]>0 - size of PSF in pixels
            % (in) lambda: 1x(size(3)+1) [double]>0 - lambdas at each pixel edge from 0
            %       to size(3)+1.  Must be monotonically increasing.
            % (out) obj - A new object
            obj@MappelHSBase(@GaussHSsMAP_Iface, imsize_,psf_sigma_);
        end
    end %public methods
    
    methods (Access=protected)
        function theta=checkTheta(obj, theta)
            if size(theta,1) ~= obj.nParams
                if length(theta)==obj.nParams
                    theta=theta';
                else
                    error('GaussHSsMAP:checkTheta', 'Invalid theta shape');
                end
            end
            ok =       all( theta(1,:) >= 0. );
            ok = ok && all( theta(1,:) <= obj.imsize(1) );
            ok = ok && all( theta(2,:) >= 0. );
            ok = ok && all( theta(2,:) <= obj.imsize(2) );
            ok = ok && all( theta(3,:) >= 0 );
            ok = ok && all( theta(3,:) <= obj.imsize(3) );
            ok = ok && all( theta(4,:) >  0. );
            ok = ok && all( theta(5,:) >= 0. );
            ok = ok && all( theta(6,:) >= 1 );
            ok = ok && all( theta(7,:) > 0. );
            if ~ok
                error('GaussHSsMAP:checkTheta', 'Invalid theta');
            end
            theta=double(theta);
        end
    end % protected methods
end % classdef
