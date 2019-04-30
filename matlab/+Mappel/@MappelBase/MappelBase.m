
classdef MappelBase < MexIFace.MexIFaceMixin
    % MappelBase.m
    %
    % Mappel base class interface for all point-emitter localization models.
    %
    % This base class implements most of the methods for each of the Mappel Models classes.  
    %
    % Mappel.MappelBase Properties
    %    ImageSize -  1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
    %    PSFSigmaMin - Minimum Gaussian point-spread function sigma size in pixels 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
    %    PSFSigmaMax - Minimum Gaussian point-spread function sigma size in pixels 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
    %    Hyperparams - Vector of hyperparameters. size:[NumHyperparams,1] 
    %    ParamNames - CellArray of model parameter names. size:[NumParams,1] These are the parameters we
    %                 estimate.
    %    HyperparamNames -  CellArray of model hyperparam names. size:[NumHyperparams,1].  These parameters control the prior distribution shape. 
    %    ParamUBound - Upper-bound for each parameter (dimension). inf=unbounded above.  Controls bounded
    %                   estimation methods.
    %    ParamLBound - Lower-bound for each parameter (dimension). -inf=unbounded below. Controls bounded estimation methods.
    %
    %
    % Mappel.MappelBase Methods
    %  * Mappel.MappelBase.samplePrior - Sample typical parameter (theta) values from the prior using hyper-parameters
    %  * Mappel.MappelBase.simulateImage - Simulate image with noise given one or more parameter (theta) values.
    %  * Mappel.MappelBase.estimate - Estimate the emitter parameters from a stack of images using maximum-likelihood. 
    %  * Mappel.MappelBase.estimatePosterior - Posterior sampling estimates for stack of images.
    %
    %   See also Mappel.MappelBase.samplePrior Mappel.MappelBase.simulateImage Mappel.MappelBase.estimate
    %   Mappel.MappelBase.estimatePosterior
    
    
    properties (Constant=true)
        MinSize = 4; %Minimum ImageSize of an image in pixels
        EstimationMethods={'Heuristic',...          %[C++(OpenMP)] Heuristic starting point guesstimate
                           'Newton',...             %[C++(OpenMP)] Newton's method with modified Cholesky step
                           'NewtonDiagonal',...     %[C++(OpenMP)] Newton's method using a diagonal hessian approximation
                           'TrustRegion',...        %[C++(OpenMP)] Trust Region method with full Hessian
                           'QuasiNewton',...        %[C++(OpenMP)] Quasi-Newton (BFGS) method with Hessian approximation
                           'SimulatedAnnealing',... %[C++(OpenMP)] Simulated Annealing global maximization method
                           'CGauss',...             %[C(OpenMP)] C Diagonal Hessian Newton implementation from Smith et. al. Nat Methods (2010)
                           'CGaussHeuristic',...    %[C(OpenMP)] C Heuristic starting guesstimate implementation from Smith et. al. Nat Methods (2010)
                           'GPUGauss',...           %[C(CUDA)] CUDA implementation from Smith et.al. Nat Methods (2010)
                           'matlab-fminsearch',...              %[MATLAB (fminsearch)] Single threaded fminsearch Nelder-Mead Simplex (derivative free) optimization from Matlab core
                           'matlab-quasi-newton',...            %[MATLAB (fminunc)] Single threaded Quasi-newton fminunc optimization from Matlab Optimization toolbox
                           'matlab-trust-region',...            %[MATLAB (fminunc)] Single threaded trust-region fminunc optimization from Matlab Optimization toolbox
                           'matlab-trust-region-reflective',... %[MATLAB (fmincon)] Single threaded trust-region optimization from Matlab Optimization toolbox
                           'matlab-interior-point'...           %[MATLAB (fmincon)] Single threaded interior-point optimization from Matlab Optimization toolbox
                           'matlab-sqp'...                      %[MATLAB (fmincon)] Single threaded sqp optimization from Matlab Optimization toolbox
                           'matlab-active-set'...               %[MATLAB (fmincon)] Single threaded active-set optimization from Matlab Optimization toolbox
                           };
        
        %List of valid algorithms for profile bounds estimation.               
        ProfileBoundsEstimationMethods={'Newton',...  %[C++(OpenMP)] [Default] Venzon and Moolgavkar implemented with Newton's method for solving nonlinear systems.
                                        'matlab-fzero' %[Matlab/C++] Use Matlab fzero to find zeros of profile likelihood as computed in C++. Single threaded.  For debugging use.
                                        };
    end

    properties (Abstract=true, Constant=true)
        ImageDim; %Dimensionality of images
    end

    properties (SetAccess = protected)
        NumParams; %Number of model parameters (i.e., model dimensionality)
        NumHyperparams; %Number of hyper-parameters, (i.e., the parameters to the model's prior)
    end %read-only properties

    %These properties have C++ storage and are accessed with getter and setter methods.
    properties
        ImageSize; % 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        PSFSigmaMin; %1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        PSFSigmaMax; %1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        Hyperparams; % [NumHyperparams,1] vector of hyper-parameters.
        ParamNames;  % [NumParams,1] cell array of parameter names
        HyperparamNames; % [NumHyperparams,1] cell array of hyper-parameter names
        ParamUBound; %Upper bound for each parameter (dimension) inf for unbounded above. Controls bounded optimization/estimation in C++.
        ParamLBound; %Lower bound for each parameter (dimension) -inf for unbounded below. Controls bounded optimization/estimation in C++.
    end

    %These properties have no C++ backing, but can be set to control the plotting and presentation of data
    properties
        ParamUnits; % [NumParams,1] cell array of parameter unit types names (as char strings)
        ParamDescription;  % [NumParams,1] cell array of parameter descriptions (as char strings)
    end

    %Default model settings can be modified by the user to customize automated options. 
    properties 
        DefaultEstimatorMethod = 'TrustRegion'; %Default optimization method for MLE/MAP estimation. Valid values are in obj.EstimationMethods.
        DefaultProfileBoundsEstimatorMethod = 'Newton'; %Default optimization method for profile bounds optimizations.  Valid values are in obj.ProfileBoundsEstimationMethods
        DefaultMCMCNumSamples = 300; % Number of final samples to use in estimation of posterior properties (mean, credible interval, cov, etc.)
        DefaultMCMCBurnin = 10; % Number of samples to throw away (burn-in) on initialization
        DefaultMCMCThin = 0; % Keep every # samples. [Value of 0 indicates use the model default. This is suggested.]
        DefaultConfidenceLevel = 0.95; % Default level at which to estimate confidence intervals must be in range (0,1).
        
        DefaultGPUGaussMLE_Iterations = 30; %GPUGaussMLE is supported for Win64 only.
        DefaultMatlabOptimizerTolerance = 1e-8; % Optimzer convergence tolerances for matlab optimization routines
        DefaultMatlabFminsearchMaxIter = 5000; % Maximum iterations for matlab fminsearch optimization routine
        DefaultMatlabToolboxMaxIter = 500; % Maximum iterations for matlab toolbox optimization routines (fminunc, fmincon)
    end

    properties (Access=protected)
        GPUGaussMLEFitType;
    end
    
    methods
        function obj = MappelBase(iface, imsize, psf_sigma_min, psf_sigma_max)
            % fixed-sigma models: obj = MappelBase(iface, imsize, psf_sigma)
            %  free-sigma models: obj = MappelBase(iface, imsize, psf_sigma_min, psf_sigma_max)
            %
            % (in) iface: The iface object MEX object name
            % (in) imsize: uint32 size:[ImageDim, 1] - size of image in pixels on each side (min: obj.MinSize)
            %
            % Further arguments depend on the model type.
            %
            % Either: (1) [for Gauss1D, and Gausss2D models with fixed PSF sigma]
            % (in) psf_sigma: size:[ImageDim, 1] >0 - size of PSF in pixels.  If scalar is passed it is copied to fill ImageDim dims.
            %
            % Or: (2) [for Gauss1Ds, Gausss2Ds, and Gausss2Dsxy models with variable PSF sigma]
            % (in) psf_sigma_min: size:[ImageDim, 1] >0 - minimum of PSF in pixels. If scalar is passed it is copied to fill ImageDim dims.
            % (in) psf_sigma_max_ratio: size:[ImageDim, 1] >1 - maximum of PSF as a ratio of psf_sigma_min
            %
            % Or: (3) [for  Gausss2Dsxy models with variable PSF sigma]
            % (in) psf_sigma_min: double size:[ImageDim, 1] >0 - minimum of PSF in pixels
            % (in) psf_sigma_max: double size:[ImageDim, 1] >psf_sigma_min - maximum of PSF in pixels

            % (out) obj - A new object
            obj = obj@MexIFace.MexIFaceMixin(iface);
            if any(imsize<obj.MinSize)
                error('MappelBase:Constructor:InvalidValue','imsize too small')
            end
            if any(psf_sigma_min<=0)
                error('MappelBase:Constructor:InvalidValue','psf_sigma must be positive')
            end
            psf_sigma_min = double(psf_sigma_min(:)');
            if numel(psf_sigma_min)<obj.ImageDim
                psf_sigma_min = [psf_sigma_min, psf_sigma_min];
            end
            imsize = uint32(imsize(:)');
            if numel(imsize) < obj.ImageDim
                imsize = [imsize, imsize];
            end
            if nargin<4
                initialized = obj.openIFace(imsize, psf_sigma_min);
                obj.PSFSigmaMin = psf_sigma_min;
                obj.PSFSigmaMax = psf_sigma_min;
            else
                if isscalar(psf_sigma_max)
                    if psf_sigma_max<=1
                        error('MappelBase:MappelBase','Got scalar psf_sigma_max <=1. Got: %f',psf_sigma_max);
                    end
                    psf_sigma_max = psf_sigma_max * psf_sigma_min;
                else
                    if ~all(psf_sigma_max>psf_sigma_min)
                        error('MappelBase:MappelBase','Got psf_sigma_max: %s must be greater than psf_sigma_min:%s',...
                            mat2str(psf_sigma_max), mat2str(psd_sigma_min));
                    end
                    psf_sigma_max = double(psf_sigma_max(:)');
                end
                    
                    
                initialized = obj.openIFace(imsize, psf_sigma_min, psf_sigma_max);
                obj.PSFSigmaMin = psf_sigma_min;
                obj.PSFSigmaMax = psf_sigma_max;
            end
            if(~initialized) 
                error('MappelBase:Constructor:LogicalError','C++ MexIFace initialization failure');
            end

            obj.ImageSize = imsize;
            obj.ParamNames = obj.call('getParamNames');
            obj.Hyperparams = obj.call('getHyperparams');
            obj.HyperparamNames = obj.call('getHyperparamNames');
            obj.NumParams = numel(obj.ParamNames);
            obj.NumHyperparams = numel(obj.Hyperparams);
            [obj.ParamLBound, obj.ParamUBound] = obj.call('getBounds');
        end
       
        function stats = getStats(obj)
            % stats = obj.getStats() - Statistics and parameters of this model
            %
            % (out) stats - 1x1 Struct describing this model.
            cstats = obj.call('getStats');
            stats = obj.convertStatsToStructs(cstats);          
        end
        
        function val = getHyperparamValue(obj, name)
            % val = obj.getHyperparamValue(name)
            %
            % Convenience method to get a hyper-parameter value by name
            %
            % (in) name: name of hyper-parameter (case insensitive)
            % (out) val: scalar value if found.
            % Throws MappelBase:HyperparamNotFound error if cannot find hyperparameter
            found = find(~cellfun(@isempty,regexpi(obj.HyperparamNames,name)),1,'first');
            if isempty(found)
                error('MappelBase:HyperparamNotFound',['Unknown hyperparameter: ', name])
            end
            val = obj.Hyperparams(found);
        end

        function val = setHyperparamValue(obj, name, val)
            % obj.setHyperparamValue(name, val)
            %
            % Convenience method to set a hyper-parameter value by name
            %
            % (in) name: name of hyper-parameter (case insensitive)
            % (in) val: new scalar value.
            % Throws MappelBase:HyperparamNotFound error if cannot find hyperparameter
            found = find(~cellfun(@isempty,regexpi(obj.HyperparamNames,name)),1,'first');
            if isempty(found)
                error('MappelBase:HyperparamNotFound',['Unknown hyperparameter: ', name])
            end
            hps = obj.Hyperparams;
            hps(found) = val;
            obj.Hyperparams = hps;
        end

        function bounded_theta = boundTheta(obj, theta)
            % bounded_theta = obj.boundTheta(theta)
            %
            % Modify thetas to esnure they remain within the interrior of the optimization domain.
            %
            % (in) theta - size:[NumParams,N] A stack of thetas bound
            % (out) bounded_theta - size:[NumParams,N] A corrected stack of thetas that are now in-bounds
            if size(theta,1) ~= obj.NumParams
                if length(theta) == obj.NumParams
                    theta = theta';
                else
                    error('MappelBase:InvalidSize', 'Invalid theta shape');
                end
            end
            bounded_theta = obj.call('boundTheta', theta);
        end
        
        function inbounds = thetaInBounds(obj, theta)
            % inbounds = thetaInBounds(theta)
            %
            % check if theta is in bounds
            %
            % (in) theta - size:[NumParams,N] A stack of thetas to check if they are in bounds
            % (out) inbounds - logical: size:[N] True for each valid theta
            if size(theta,1) ~= obj.NumParams
                if length(theta) == obj.NumParams
                    theta=theta';
                else
                    error('MappelBase:checkTheta', 'Invalid theta shape');
                end
            end
            inbounds = obj.call('thetaInBounds', theta);
        end

        function theta = samplePrior(obj, count)
            % theta = obj.samplePrior(count)
            %
            % sample form the built-in prior using current obj.Hyperparams
            %
            % (in) count: int (default 1) number of thetas to sample
            % (out) theta: A (NumParams X n) double of theta values
            if nargin==1
                count = 1;
            end
            count = obj.checkCount(count);
            theta = obj.call('samplePrior', count);
        end

        function image = modelImage(obj, theta)
            % image = obj.modelImage(theta)
            %
            % (in) theta: size:[NumParams, n] double of theta values
            % (out) image: size:[flip(ImageSize), n] image stack
            theta=obj.checkTheta(theta);
            image= obj.call('modelImage', theta);
        end
        
        function image = modelDipImage(obj, theta)
            % image = obj.modelDipImage(theta)
            %
            % This just calls obj.modelImage(theta), then rotates the image so that x
            % goes from left to right, and returns a DIP image (or image
            % sequence.)
            %
            % (in) theta: size:[NumParams, n] double of theta values
            % (out) image: size:[flip(ImageSize), n] DIP image stack of model (mean) value at each pixel.
            image = dip_image(obj.modelImage(theta));
        end

        function image = simulateImage(obj, theta, count)
            % image = obj.simulateImage(theta, count)
            %
            % If theta is size:[NumParams,1] then count images with that theta are
            % simulated.  Default count is 1.  If theta is size:[NumParams, n] with n>1
            % then n images are simulated, each with a separate theta, and count is ignored.
            % The noise model of the Model class is used to add noise to the sampled model images.
            %
            % (in) theta: size:[NumParams, 1] or [NumParams, n] double theta value.
            % (in) count: [optional] the number of independent images to generate. Only used if theta is a single parameter.
            % (out) image: a stack of n images, all sampled with parameters theta.
            theta=obj.checkTheta(theta);
            if nargin==2
                count = 1;
            elseif nargin==3
                if(count>1 && size(theta,2)>1)
                    error('MappelBase:simulateImage', 'Cannot set count>1 when theta_stack has more than one theta');
                end
            end
                    
            count = obj.checkCount(count);
            image = obj.call('simulateImage', theta, count);
        end

        function image = simulateDipImage(obj, varargin)
            % image = obj.simulateDipImage(theta, count)
            %
            % Requires: dip_image package. This just calls obj.simulateImage(theta), then rotates the
            % image so that x goes from left to right, and returns a DIP image (or image sequence.)
            %
            % (in) theta: size:[NumParams, 1] or [NumParams, n] double theta value.
            % (in) count: [optional] the number of independent images to generate. 
            %             Only used if theta is a single parameter.
            % (out) image: a dim_images object holding a stack of n images all sampled with parameters theta
            image = dip_image(obj.simulateImage(varargin{:}));
        end

        function llh = modelLLH(obj, image, theta)
            % llh = obj.modelLLH(image, theta)
            %
            % Compute the log-likelihood of the images a the given thetas. This takes in a N
            % images and M thetas.  If M=N=1, then we return a single LLH.  If there are N=1 images and
            % M>1 thetas, we return M LLHs of the same image with each of the thetas.  Otherwise, if
            % there is M=1 thetas and N>1 images, then we return N LLHs for each of the images given
            % theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) llh: size:[1,max(M,N)] double of log-likelihoods
            theta = obj.checkTheta(theta);
            image = obj.checkImage(image);
            if size(image,obj.ImageDim+1) ~= size(theta,2) && size(theta,2)~=1 && ~ismatrix(image)
                error('MappelBase:LLH','image and theta must match dims');
            end
            llh = obj.call('modelLLH', image, theta);
        end
        
        function rllh = modelRLLH(obj, image, theta)
            % rllh = obj.modelRLLH(image, theta)
            %
            % Compute the relative-log-likelihood of the images a the given thetas. This takes in a N
            % images and M thetas.  If M=N=1, then we return a single LLH.  If there are N=1 images and
            % M>1 thetas, we return M LLHs of the same image with each of the thetas.  Otherwise, if
            % there is M=1 thetas and N>1 images, then we return N LLHs for each of the images given
            % theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) llh: size:[1,max(M,N)] double of relative log-likelihoods
            theta = obj.checkTheta(theta);
            image = obj.checkImage(image);
            if size(image,obj.ImageDim+1) ~= size(theta,2) && size(theta,2)~=1 && ~ismatrix(image)
                error('MappelBase:RLLH','image and theta must match dims');
            end
            rllh = obj.call('modelRLLH', image, theta);
        end
        
        function grad = modelGrad(obj, image, theta)
            % grad = obj.modelGrad(image, theta) - Compute the model gradient.
            %
            % This takes in a N images and M thetas.  If M=N=1, then we return a single Grad.  If there
            % are N=1 images and M>1 thetas, we return M Grads of the same image with each of the thetas.
            % Otherwise, if there is M=1 thetas and N>1 images, then we return N Grads for each of the
            % images given theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) grad: a (NumParams X max(M,N)) double of gradient vectors
            theta = obj.checkTheta(theta);
            image = obj.checkImage(image);
            if size(image,obj.ImageDim+1) ~= size(theta,2)  && size(theta,2)~=1 && ~ismatrix(image)
                error('MappelBase:modelGrad','image and theta must match dims');
            end
            grad = obj.call('modelGrad', image, theta);
        end

        function hess = modelHessian(obj, image, theta)
            % hess = obj.modelHessian(image, theta) - Compute the model hessian
            %
            % This takes in a N images and M thetas.  If M=N=1, then we return a single Hessian.  If
            % there are N=1 images and M>1 thetas, we return M Hessian of the same image with each of the
            % thetas.  Otherwise, if there is M=1 thetas and N>1 images, then we return N Hessians for
            % each of the images given theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) hess: a (NumParams X NumParams X max(M,N)) double of hessian matrices
            theta = obj.checkTheta(theta);
            image = obj.checkImage(image);
            if size(image,obj.ImageDim+1) ~= size(theta,2)  && size(theta,2)~=1 && ~ismatrix(image)
                error('MappelBase:modelHessian','image and theta must match dims');
            end
            hess = obj.call('modelHessian', image, theta);
        end

        function definite_hess = modelHessianNegativeDefinite(obj, image, theta)
            % definite_hess = obj.modelHessianNegativeDefinite(image, theta)
            %
            % Compute a negative definite approximation to the model hessian at theta. This takes in a N
            % images and M thetas.  If M=N=1, then we return a single Hessian.  If there are N=1 images
            % and M>1 thetas, we return M Hessian of the same image with each of the thetas.  Otherwise,
            % if there is M=1 thetas and N>1 images, then we return N Hessians for each of the images
            % given theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) hess: a (NumParams X NumParams X max(M,N)) double of hessian matrices
            definite_hess = obj.modelHessian(image,theta);
            definite_hess = obj.callstatic('negativeDefiniteCholeskyApprox', definite_hess);
        end

        function varargout = modelObjective(obj, image, theta, negate)
            % [rllh, grad, hess, definite_hess, llh] = obj.modelObjective(image, theta, negate)
            %
            % The model objective is simply the log-likelihood for MLE models, and the sum of the
            % log-likelihood an log-prior for MAP models.
            %
            % A convenience function for objective based optimization.  Works on a single image and
            % single theta and shares the stencil to compute the LLH,Grad,Hessian as the 3 default
            % outputs, with a Negative(Positive) definite approximation to the hessian as the 4th output
            % argument.  All output arguments after the first are optional and not requesting them will
            % cause the C++ code to not compute them at all.
            %
            % Calling all these functions in one group allows faster use with matlab optimization. Also
            % we omit any explicit checking of inputs to speed up calls to objectives.
            %
            % These models treat maximum likelihood estimation as a maximization problem, so LLH, Grad,
            % Hessian are all defined for a maximization problem where the hessian will be negative
            % definite around the maximum.  To convert to the inverse minimization problem set the negate
            % flag.  This flag also affects weather the 4th argument is negative- or positive-definite.
            %
            % (in) image: an image, double size:[flip(ImageSize)]
            % (in) theta: a parameter value size:[NumParams,1] double of theta
            % (in) negate: optional) boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh: relative log likelihood scalar double
            % (out) grad: [optional] grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: [optional] hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: [optional] negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
            % (out) llh:  full log likelihood with constant terms as a scalar double
            if nargin<4
                negate = false;
            end
            [varargout{1:nargout}] = obj.call('modelObjective', image, theta, negate);
        end

        function varargout = modelObjectiveAPosteriori(obj, image, theta, negate)
            % [rllh, grad, hess, definite_hess] = obj.modelObjectiveAPosteriori(image, theta, negate)
            %
            % This is always an a-posteriori (MAP) objective irrespective of the model type (MLE / MAP).
            %
            % A convenience function for objective based optimization.  Works on a single image and
            % single theta and shares the stencil to compute the LLH,Grad,Hessian as the 3 default
            % outputs, with a Negative(Positive) definite approximation to the hessian as the 4th output
            % argument.  All output arguments after the first are optional and not requesting them will
            % cause the C++ code to not compute them at all.
            %
            % Calling all these functions in one group allows faster use with matlab optimization. Also
            % we omit any explicit checking of inputs to speed up calls to objectives.
            %
            % These models treat maximum likelihood estimation as a maximization problem, so LLH, Grad,
            % Hessian are all defined for a maximization problem where the hessian will be negative
            % definite around the maximum.  To convert to the inverse minimization problem set the negate
            % flag.  This flag also affects weather the 4th argument is negative- or positive-definite.
            %
            % (in) image: an image, double size:[flip(ImageSize)]
            % (in) theta: a parameter value size:[NumParams,1] double of theta
            % (in) negate: [optional] boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh: relative log likelihood scalar double
            % (out) grad: [optional] grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: [optional] hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: [optional]  negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
            % (out) llh:  full log likelihood with constant terms as a scalar double
            if nargin<4
                negate = false;
            end
            [varargout{1:nargout}] = obj.call('modelObjectiveAPosteriori', image, theta, negate);
        end

        function varargout = modelObjectiveLikelihood(obj, image, theta, negate)
            % [rllh, grad, hess, definite_hess] = obj.modelObjectiveLikelihood(image, theta, negate)
            %
            % This is always a pure likelihood (MLE) objective irrespective of the model type (MLE /
            % MAP).
            %
            % A convenience function for objective based optimization.  Works on a single image and
            % single theta and shares the stencil to compute the LLH,Grad,Hessian as the 3 default
            % outputs, with a Negative(Positive) definite approximation to the hessian as the 4th output
            % argument.  All output arguments after the first are optional and not requesting them will
            % cause the C++ code to not compute them at all.
            %
            % Calling all these functions in one group allows faster use with matlab optimization. Also
            % we omit any explicit checking of inputs to speed up calls to objectives.
            %
            % These models treat maximum likelihood estimation as a maximization problem, so LLH, Grad,
            % Hessian are all defined for a maximization problem where the hessian will be negative
            % definite around the maximum.  To convert to the inverse minimization problem set the negate
            % flag.  This flag also affects weather the 4th argument is negative- or positive-definite.
            %
            % (in) image: an image, double size:[flip(ImageSize)]
            % (in) theta: a parameter value size:[NumParams,1] double of theta
            % (in) negate: [optional] boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh: relative log likelihood scalar double
            % (out) grad: [optional] grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: [optional] hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: [optional] negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
            % (out) llh:  full log likelihood with constant terms as a scalar double
            if nargin<4
                negate = false;
            end
            [varargout{1:nargout}] = obj.call('modelObjectiveLikelihood', image, theta, negate);
        end

        function varargout = modelObjectivePrior(obj, theta, negate)
            % [rllh, grad, hess, definite_hess] = obj.modelObjectiveLPrior(theta, negate)
            %
            % This is always pure prior likelihood objective irrespective of the model type (MLE / MAP).
            % The prior does not depend on the image data.
            %
            % A convenience function for objective based optimization.  Works on a single image and
            % single theta and shares the stencil to compute the LLH,Grad,Hessian as the 3 default
            % outputs, with a Negative(Positive) definite approximation to the hessian as the 4th output
            % argument.  All output arguments after the first are optional and not requesting them will
            % cause the C++ code to not compute them at all.
            %
            % Calling all these functions in one group allows faster use with matlab optimization. Also
            % we omit any explicit checking of inputs to speed up calls to objectives.
            %
            % These models treat Maximum Likelihood estimation as a maximization problem, so LLH, Grad,
            % Hessian are all defined for a maximization problem where the hessian will be negative
            % definite around the maximum.  To convert to the inverse minimization problem set the negate
            % flag.  This flag also affects weather the 4th argument is negative- or positive-definite.
            %
            % (in) theta: a parameter value size:[NumParams,1] double of theta
            % (in) negate: [optional] boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh:  relative log likelihood scalar double
            % (out) grad: [optional] grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: [optional] hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: [optional] negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
            % (out) llh:  full log likelihood with constant terms as a scalar double
            if nargin<3
                negate = false;
            end
            [varargout{1:nargout}] = obj.call('modelObjectivePrior', theta, negate);
        end

        function varargout = modelObjectiveComponents(obj, image, theta)
            % [llh_components, grad_components, hess_components] = obj.modelObjectiveComponents(image, theta)
            %
            % [DEBUGGING]
            %  Component-wise break down of model objective into individual contributions from pixels and
            %  model components. Each pixel that is not NAN (as well as the prior for MAP models) will
            %  contribute linearly to the overall log-likelihood objective. Because their probabilities
            %  are multiplied in the model, their log-likelihoods are summed.  Here each individual pixel
            %  and the prior will have their individual values returned
            % NumComponets is prod(ImageSize) for MLE models and prod(ImageSize)+NumParams for MAP models
            % where each of the final components corresponds to a single parameter in the
            %
            % (in) image: an image, double size:[flip(ImageSize)]
            % (in) theta: a parameter value size:[NumParams,1] double of theta
            % (out) llh: log likelihood components size:[1,NumComponents]
            % (out) grad: [optional] components of grad of log likelihood size:[NumParams,NumComponents*]  
            %             * there is only a single extra grad component added for prior in MAP models.
            % (out) hess: [optional] hessian of log likelihood double size:[NumParams,NumParams,NumComponents*] 
            %             * there is only a single extra hess component added for prior in MAP models.
            [varargout{1:nargout}] = obj.call('modelObjectiveComponents', image, theta);
        end

        function fisherI = expectedInformation(obj, theta)
            % fisherI = obj.expectedInformation(theta) -
            % Compute the Expected Information at theta (aka., Fisher Information matrix)
            %
            % (in) theta: an size:[NumParams, N] double of theta values
            % (out) fisherI: a  size:[NumParams, Numparams, N] matrix of the fisher information at each theta
            theta = obj.checkTheta(theta);
            fisherI = obj.call('expectedInformation', theta);
        end

        function crlb = CRLB(obj, theta)
            % crlb = obj.CRLB(theta) - Compute the Cramer-Rao Lower Bound at theta
            %
            % computed as the diagonal of the inverse of the fisher information, I(theta).  I(theta)
            % should be positive definite at the maximum. The CRLB is a lower bounds on the variance at
            % theta for any unbiased estimator.
            %
            % (in) theta: size:[NumParams, n] stack of theta values
            % (out) crlb: size:[NumParams, n] stack of the Cramer-Rao lower bound on the variances of each parameter.
            theta = obj.checkTheta(theta);
            crlb = obj.call('CRLB', theta);
        end

        function obsI = observedInformation(obj, image, theta_mle)
            % obsI = obj.ObservedInformation(image, theta_mle) - Compute the observed information matrix at theta.
            %
            % Should only be called at theta=theta_mle, i.e., at the maximum (mode) of the distribution.
            % ObsI should be positive definite at the mode.
            % This is equivalent to the negative hessian of the log-likelihood.
            %
            % (in) image: a single image or a stack of images
            % (in) theta_mle: an estimated  MLE theta size:[NumParams,N].  One theta_mle per image.
            % (out) obsI: the observed information matrix size:[NumParams, NumParams]
            image = obj.checkImage(image);
            theta_mle = obj.checkTheta(theta_mle);
            obsI = -obj.call('modelHessian',image, theta_mle);
        end

        function [observed_lb, observed_ub] = errorBoundsObserved(obj, image, theta_mle, confidence, obsI)
            % [observed_lb, observed_ub] = obj.errorBoundsObserved(image, theta_mle, confidence, obsI)
            %
            % Compute the error bounds using the observed information at the MLE estimate theta_mle.
            %
            % (in) image: a single image or a stack of n images
            % (in) theta_mle: the MLE estimated theta size:[NumParams,n]
            % (in) confidence: [optional] desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (in) obsI: [optional] observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n]
            % (out) observed_lb: the observed-information-based confidence interval lower bound for parameters, size:[NumParams,n]
            % (out) observed_ub: the observed-information-based confidence interval upper bound for parameters, size:[NumParams,n]
            image = obj.checkImage(image);
            theta_mle = obj.checkTheta(theta_mle);
            if nargin<4
                confidence = obj.DefaultConfidenceLevel;
            end
            if nargin<5
                [observed_lb, observed_ub] = obj.call('errorBoundsObserved',image, theta_mle, confidence);
            else
                [observed_lb, observed_ub] = obj.call('errorBoundsObserved',image, theta_mle, confidence, obsI);
            end
        end

        function [expected_lb, expected_ub] = errorBoundsExpected(obj, theta_mle, confidence)
            % [expected_lb, expected_ub] = obj.errorBoundsExpected(theta_mle, confidence)
            %
            % Compute the error bounds using the expected (Fisher) information at the MLE estimate.  This
            % is independent of the image, assuming Gaussian error with variance given by CRLB.
            %
            % (in) theta_mle: the theta MLE's to estimate error bounds for. size:[NumParams,N]
            % (in) confidence: [optional]  desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (out) expected_lb: the expected-information-based confidence interval lower bound for parameters for each theta, size:[NumParams,N]
            % (out) expected_ub: the expected-information-based confidence interval upper bound for parameters for each theta, size:[NumParams,N]
            theta_mle = obj.checkTheta(theta_mle);
            if nargin<3
                confidence = obj.DefaultConfidenceLevel;
            end
            [expected_lb, expected_ub] = obj.call('errorBoundsExpected', theta_mle, confidence);
        end

        function varargout = errorBoundsProfileLikelihood(obj, image, theta_mle, confidence, theta_mle_rllh, obsI, algorithm, estimated_idxs)
            % [profile_lb, profile_ub, stats]
            %    = obj.errorBoundsProfileLikelihood(images, theta_mle, confidence, theta_mle_rllh, obsI, estimate_parameters, algorithm)
            %
            % Compute the profile log-likelihood bounds for a stack of images, estimating upper and lower bounds for each requested parameter.
            % Uses the Venzon and Moolgavkar (1988) algorithm, implemented in OpenMP.
            %
            % (in) images: a stack of N images
            % (in) theta_mle: a stack of N theta_mle estimates corresponding to the image size
            % (in) confidence: [optional] desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (in) theta_mle_rllh: [optional] size:[N] relative-log-likelihood at each image's theta_mle.  Otherwise it must be re-computed. 
            % (in) theta_rllh: [optional] 
            % (in) obsI: [optional] observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n].
            %                      Must be recomputed if not provided or provided as empty array.
            % (in) algorithm: [optional] [Default = obj.DefaultProfileBoundsEstimatorMethod] 
            %                             Valid methods are in obj.ProfileBoundsEstimationMethods
            % (in) estimated_idxs: [optional] indexs of estimated parameters.  Empty array (defulat) indicates
            %                       estimate all parameters.
            % (out) profile_lb: size [NumParams,N] lower bounds for each parameter to be estimated. NaN if parameter was not estimated
            % (out) profile_ub: size [NumParams,N] upper bounds for each parameter to be estimated. NaN if parameter was not estimated
            % (out) profile_points: [optional] size[NumParams,2,NumParams,N] Profile maxima thetas at which
            %           profile bounds were obtained.  Each [NumParams,2,NumParams] slice are the thetas found defining the 
            %           the lower and upper bound for each parameter in sequence as the 3-rd dimension.
            %           The 4-th dimension is used if the profile is run on multiple images.  These can
            %           be useful to test for the quality of the estimated points.
            % (out) profile_points_rllh: [optional] size [2,NumParams,N], rllh at each returne profile_point.
            % (out) stats: [optional] struct of fitting statistics.
            image = obj.checkImage(image);
            if nargin<3
                [theta_mle, theta_mle_rllh, obsI] = obj.estimate(image);
            else
                theta_mle = obj.checkTheta(theta_mle);
                if nargin<5
                    theta_mle_rllh=[];
                end
                if nargin<6
                    obsI=[];
                end
            end
            if nargin<4
                confidence = obj.DefaultConfidenceLevel;
            end
            if nargin<7
                algorithm = obj.DefaultProfileBoundsEstimatorMethod;
            end
            if nargin<8
                estimated_idxs = uint64(1:obj.NumParams);
            elseif numel(unique(estimated_idxs))~=numel(estimated_idxs)
                error('Mappel:errorBoundsProfileLikelihood','estimated_idxs must be unique.  Got: %s',mat2str(estimated_idxs))
            elseif any(estimated_idxs<1) || any(estimated_idxs>obj.NumParams)
                error('Mappel:errorBoundsProfileLikelihood','estimated_idxs must be between 1 and obj.NumParams.  Got: %s',mat2str(estimated_idxs))    
            else
                estimated_idxs = uint64(sort(estimated_idxs));
            end
                
            switch lower(algorithm)
                case {'n','newt','newton'}
                    [varargout{1:nargout}] = obj.call('errorBoundsProfileLikelihood',image, theta_mle, confidence, theta_mle_rllh, obsI, estimated_idxs-1);
                case {'matlab-fzero','fzero'}
                    [varargout{1:nargout}] = obj.errorBoundsProfileLikelihood_matlab_fzero(obj, image, theta_mle, confidence, theta_mle_rllh, obsI, estimated_idxs);
                case {'matlab-fsolve','fsolve'}
                    [varargout{1:nargout}] = obj.errorBoundsProfileLikelihood_matlab_fsolve(obj, image, theta_mle, confidence, theta_mle_rllh, obsI, estimated_idxs);
                otherwise
                    error('Mappel:errorBoundsProfileLikelihood','Unknown algorithm %s',algorithm);
            end
        end

        function varargout = errorBoundsProfileLikelihoodDebug(obj, image, theta_mle, theta_mle_rllh, obsI, estimate_parameter, llh_delta, algorithm)
            % [profile_lb, profile_ub, seq_lb, seq_ub, seq_lb_rllh, seq_ub_rllh, stats]
            %    = obj.errorBoundsProfileLikelihoodDebug(image, estimate_parameter, theta_mle, theta_mle_rllh, obsI, llh_delta)
            %
            % [DEBUGGING]
            % Compute the profile log-likelihood bounds for a single images , estimating upper and lower bounds for each requested  parameter.
            % Uses the Venzon and Moolgavkar (1988) algorithm.
            %
            % (in) image: a single images
            % (in) theta_mle:  theta ML estimate
            % (in) theta_mle_rllh: relative-log-likelihood at image.
            % (in) obsI: a observed fisher information matrix at theta_mle
            % (in) estimate_parameter: integer index of parameter to estimate size:[NumParams]
            % (in) llh_delta: [optional] Negative number, indicating LLH change from maximum at the profile likelihood boundaries.
            %                  [default: confidence=0.95; llh_delta = -chi2inv(confidence,1)/2;]
            % (out) profile_lb:  scalar lower bound for parameter
            % (out) profile_ub:  scalar upper bound for parameter
            % (out) seq_lb: size:[NumParams,Nseq_lb]  Sequence of Nseq_lb points resulting from VM algorithm for lower bound estimate
            % (out) seq_ub: size:[NumParams,Nseq_ub]  Sequence of Nseq_ub points resulting from VM algorithm for upper bound estimate
            % (out) seq_lb_rllh: size:[Nseq_lb]  Sequence of RLLH at each of the seq_lb points
            % (out) seq_ub_rllh: size:[Nseq_ub]  Sequence of RLLH at each of the seq_ub points
            % (out) stats: struct of fitting statistics.
            image = obj.checkImage(image);
            if nargin<3
                [theta_mle, theta_mle_rllh, obsI] = obj.estimate(image);
            else
                theta_mle = obj.checkTheta(theta_mle);
                if nargin<5 || isempty(theta_mle_rllh)
                    theta_mle_rllh=-inf;
                end
                if nargin<6
                    obsI=[];
                end
            end
            if nargin<6
                estimate_parameter=1;
            elseif ~isscalar(estimate_parameter)
                error('Mappel:errorBoundsProfileLikelihoodDebug','estimate_parameter must be scalar');                
            elseif ~(estimate_parameter>=1 && estimate_parameter<=obj.NumParams)
                error('Mappel:errorBoundsProfileLikelihoodDebug','estimate_parameter must be between 1 and obj.NumParams.  Got: %i',estimate_parameter);
            end
            if nargin<7
                llh_delta = -chi2inv(obj.DefaultConfidenceLevel,1)/2;
            elseif llh_delta>=0
                error('Mappel:errorBoundsProfileLikelihoodDebug','llh_delta should be negative got: %f',llh_delta);
            end
            if nargin<8
                algorithm = obj.DefaultProfileBoundsEstimatorMethod;
            end
            switch lower(algorithm)
                case {'n','newt','newton'}
                    [varargout{1:nargout}] = obj.call('errorBoundsProfileLikelihoodDebug',image, theta_mle, theta_mle_rllh, obsI, uint64(estimate_parameter)-1, llh_delta);
                case {'matlab-fzero','fzero'}
                    [varargout{1:nargout}] = obj.errorBoundsProfileLikelihoodDebug_matlab_fzero(obj, image, theta_mle_rllh, obsI, estimated_parameter-1, confidence, theta_mle_rllh, obsI, estimate_parameters);
                otherwise
                    error('Mappel:errorBoundsProfileLikelihood','Unknown algorithm %s',algorithm);
            end
            
        end

        function varargout = errorBoundsPosteriorCredible(obj, image, theta_mle_approx, confidence, num_samples, burnin, thin)
            % [credible_lb, credible_ub, posterior_sample] = obj.errorBoundsPosteriorCredible(image, theta_mle_approx, confidence, max_samples)
            %
            % Compute the error bounds using the posterior credible interval, estimated with an MCMC sampling, using at most max_samples
            %
            % (in) image: a stack of n images to estimate
            % (in) theta_mle_approx: [optional] Initial guesses for the MLE.  This is just used to
            %                   initialize the chain, and need not be precise.
            %                   size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated, valid parameter values will be kept even if invalid ones require initialization.
            % (in) confidence: [optional] desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
            % (in) num_samples: [optional] Number of (post-filtering) posterior samples to acquire. [default=obj.DefaultMCMCNumSamples]
            % (in) burnin: [optional] Number of samples to throw away (burn-in) on initialization [default=obj.DefaultMCMCBurnin]
            % (in) thin: [optional] Keep every # samples. Value of 0 indicates use the model default. This is suggested.
            %                       When thin=1 there is no thinning.  This is also a good option if rejections are rare. [default=obj.DefaultMCMCThin]
            % (out) posterior_mean: [optional] size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            % (out) credible_lb: [optional] size:[NumParams,n] posterior credible interval lower bounds for each parameter for each image
            % (out) credible_ub: [optional] size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            [varargout{1:nargout}] = obj.estimatePosterior(image, theta_mle, confidence, num_samples, burnin, thin);
        end

%         function  coverageProbability(obj, theta, Nsamples, 
        
        function [theta, varargout] = estimateMax(obj, image, estimator_algorithm, theta_init)
            % [theta, obsI, llh, stats] = obj.estimateMax(image, estimator_algorithm, theta_init)
            %
            % Use a multivariate constrained optimization algorithm to estimate the model parameter theta for each of
            % a stack of n images.  Returns observed information and log-likelihood as optional parameters.
            %
            % This is the maximum-likelihood estimate for 'MLE' models and the maximum a posteriori estimate for 'MAP' models.
            %
            % 'TrustRegion' is suggested as the most robust estimator.
            %
            % (in) image: a stack of N images
            % (in) estimator_algorithm: [optional] name from obj.EstimationMethods.  The optimization method. [default=DefaultEstimatorMethod]
            % (in) theta_init: [optional] Initial theta guesses size:[NumParams, N].  Values of 0 indicate
            %            that we have no initial guess and the estimator should form its own guess.
            % (out) theta: size:[NumParams, N] estimated theta maximum for each image.
            % (out) rllh: [optional] size:[1,N] double of the relative log likelihood at each theta estimate.
            % (out) obsI: [optional] size:[NumParams,NumParams, N] the observed information at the MLE for each image.
            % (out) stats: [optional] A 1x1 struct of fitting statistics.
            image = obj.checkImage(image);
            if nargin<3
                estimator_algorithm = obj.DefaultEstimatorMethod;
            else
                estimator_algorithm = convertStringsToChars(estimator_algorithm);
            end
            if nargin<4
                theta_init = [];
            end
            %Check to make sure we have a theta_init for each image
            nIms = size(image, obj.ImageDim+1);
            theta_init = obj.checkThetaInit(theta_init, nIms);

            if ~ischar(estimator_algorithm)
                error('MappelBase:estimate', 'Invalid estimation method name');
            end
            switch lower(estimator_algorithm)
                case 'gpugauss'
                    [theta, varargout{1:nargout-1}] = obj.estimate_GPUGaussMLE(image);
                case {'fminsearch', 'matlab-fminsearch'}
                    [theta, varargout{1:nargout-1}] = obj.estimate_fminsearch(image, theta_init);
                case {'matlab-quasi-newton','matlab-quasi','matlab-reflective','matlab-trust-region-reflective','matlab-trust','matlab-trust-region','matlab-interior-point','matlab-interior','matlab-active-set','matlab-sqp'}
                    [theta, varargout{1:nargout-1}] = obj.estimate_toolbox(image, theta_init, estimator_algorithm(8:end));
                case {'matlabquasinewton','matlabquasi','matlabreflective','matlabtrustregionreflective','matlabtrust','matlabtrustregion','matlabinteriorpoint','matlabinterior','matlabsqp','matlabactiveset'}
                    [theta, varargout{1:nargout-1}] = obj.estimate_toolbox(image, theta_init, estimator_algorithm(7:end));
                case {'trust','tr'}
                    [theta, varargout{1:nargout-1}] = obj.call('estimateMax',image, 'TrustRegion', theta_init);
                case {'newt','n'}
                    [theta, varargout{1:nargout-1}] = obj.call('estimateMax',image, 'Newton', theta_init);
                case {'qn','quasi','bfgs'}
                    [theta, varargout{1:nargout-1}] = obj.call('estimateMax',image, 'QuasiNewton', theta_init);
                case {'nd','diag','diagonal'}
                    [theta, varargout{1:nargout-1}] = obj.call('estimateMax',image, 'NewtonDiagonal', theta_init);
                otherwise
                    [theta, varargout{1:nargout-1}] = obj.call('estimateMax',image, estimator_algorithm, theta_init);
            end
        end

        function varargout = estimateProfileLikelihood(obj, image, fixed_parameters, fixed_values, estimator_algorithm, theta_init)
            % [profile_likelihood, profile_parameters, stats]  = obj.estimateProfileLikelihood(image, fixed_parameters, fixed_values, estimator_algorithm, theta_init)
            %
            % Compute the profile likelihood for a single image and single parameter, over a range of
            % values.  For each value, the parameter of interest is fixed and the other parameters are
            % optimized with the estimator_algorithm in parallel with OpenMP.
            %
            % At least one parameter must be fixed and at least one parameter must be free.
            %
            % (in) image: a single images
            % (in) fixed_parameters: uint64 size:[NParams,1] mask 0=free 1=fixed.  
            %                  [or] if numel(fixed_parameters)<NParams, it is treated as a vector of
            %                        indexes of fixed parameters (uses matlab 1-based indexing).
            % (in) fixed_values: size:[NumFixedParams,N], a vector of N values for each of the fixed parameters at which to maximize the other (free) parameters at.
            % (in) estimator_algorithm: [optional] name for the optimization method. (default = 'TrustRegion') [see: obj.EstimationMethods]
            % (in) theta_init: [optional] Initial theta guesses size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                   If only a single parameter [NumParams,1] is given, each profile estimation will use this single theta_init.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated, valid parameter values will be kept even if invalid ones require initialization.
            % (out) profile_likelihood: size:[N,1] profile likelihood for the parameter at each value.,
            % (out) profile_parameters: [optional] size:[NumParams,N] parameters that achieve the profile likelihood maximum at each value.
            % (out) stats: [optional] Estimator stats dictionary.
            image = obj.checkImage(image);
            if nargin<5
                estimator_algorithm = obj.DefaultEstimatorMethod;
            end
            if nargin<6
                theta_init = [];
            end
            if isempty(fixed_parameters)
                error('MappelBase:estimateProfileLikelihood','No fixed parameters given.');
            elseif numel(fixed_parameters)==obj.NumParams
                fixed_idxs = uint64(find(fixed_parameters));
            else
                fixed_idxs = uint64(fixed_parameters);
            end
            Nfixed = numel(fixed_idxs);
            if Nfixed<1 || Nfixed>obj.NumParams
                error('MappelBase:InvalidValue','fixed_parameters should have at least one fixed and at least one free parameter');
            end
            if isvector(fixed_values)
                if Nfixed==1
                    fixed_values = fixed_values(:)';
                else
                    fixed_values = fixed_values(:);
                end
            end
            fixed_values_sz = size(fixed_values);
            if fixed_values_sz(1) ~= Nfixed
                error('MappelBase:InvalidSize','fixed_values must have one row for each parameter indicated in fixed_parameters');
            end
            theta_init = obj.checkThetaInit(theta_init, fixed_values_sz(2)); %expand theta_init to size:[NumParams,n]

            [varargout{1:nargout}] = obj.call('estimateProfileLikelihood',image, fixed_idxs-1, fixed_values, estimator_algorithm, theta_init);
        end

        function varargout = estimateMaxDebug(obj, image, estimator_algorithm, theta_init)
            % [theta, obsI, llh, sample, sample_rllh, stats] = obj.estimateMaxDebug(image, estimator_algorithm, theta_init)
            %
            % DEBUGGING]
            % Estimate for a single image.  Returns entire sequence of evaluated points and their LLH.
            % The first entry of the evaluated_seq is theta_init.  The last entry may or may not be
            % theta_est.  It is strictly a sequence of evaluated thetas so that the length of the
            % evaluated_seq is the same as the number of RLLH evaluations performed by the maximization
            % algorithm.
            %
            % (in) image: a size:[flip(ImageSize)] image
            % (in) estimator_algorithm: [optional] name from obj.EstimationMethods.  The optimization method. [default=DefaultEstimatorMethod]
            % (in) theta_init: [optional] Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (out) theta: size:[NumParams,1] estimated theta value
            % (out) rllh: a (1 X 1) double of the relative log likelihood at each theta estimate.
            % (out) obsI:  size:[NumParams,NumParams] the observed information at theta
            % (out) evaluated_seq: A (NumParams X n) sequence of parameter values at which the model was
            %                      evaluated in the course of the maximization algorithm.
            % (out) evaluated_seq_rllh: A (1 X n) array of relative log likelihoods at each evaluated theta
            % (out) stats: A 1x1 struct of fitting statistics.
            image=obj.checkImage(image);
            if nargin==2
                estimator_algorithm='Newton';
            end
            if nargin<4
                theta_init=[];
            end
            %Check to make sure we have a theta_init for each image
            theta_init=obj.checkThetaInit(theta_init, 1);
            
            if ~ischar(estimator_algorithm)
                error('MappelBase:estimateDebug', 'Invalid estimation method name');
            end
            switch estimator_algorithm
                case 'GPUGauss'
                    [varargout{1:nargout}] = obj.estimateDebug_GPUGaussMLE(image);
                case {'fminsearch', 'matlab-fminsearch'}
                    [varargout{1:nargout}] = obj.estimateDebug_fminsearch(image, theta_init);
                case {'matlab-quasi-newton','matlab-quasi','matlab-reflective','matlab-trust-region-reflective','matlab-trust','matlab-trust-region','matlab-interior-point','matlab-interior','matlab-active-set','matlab-sqp'}
                    [varargout{1:nargout}] = obj.estimateDebug_toolbox(image, theta_init, estimator_algorithm(8:end));
                case {'matlabquasinewton','matlabquasi','matlabreflective','matlabtrustregionreflective','matlabtrust','matlabtrustregion','matlabinteriorpoint','matlabinterior','matlabsqp','matlabactiveset'}
                    [varargout{1:nargout}] = obj.estimateDebug_toolbox(image, theta_init, estimator_algorithm(7:end));
                case {'trust','tr'}
                    [varargout{1:nargout}] = obj.call('estimateMaxDebug',image, 'TrustRegion', theta_init);
                case {'newt','n'}
                    [varargout{1:nargout}] = obj.call('estimateMaxDebug',image, 'Newton', theta_init);
                case {'qn','quasi','bfgs'}
                    [varargout{1:nargout}] = obj.call('estimateMaxDebug',image, 'QuasiNewton', theta_init);
                case {'nd','diag','diagonal'}
                    [varargout{1:nargout}] = obj.call('estimateMaxDebug',image, 'NewtonDiagonal', theta_init);
                otherwise
                    [varargout{1:nargout}] = obj.call('estimateMaxDebug',image, estimator_algorithm, theta_init);
            end
            if nargout==6
                varargout{6} = MexIFace.MexIFaceMixin.convertStatsToStructs(varargout{6});
            end
        end    

        function varargout = estimatePosterior(obj, image, theta_init, confidence, num_samples, burnin, thin)
            % [posterior_mean, credible_lb, credible_ub, posterior_cov, mcmc_samples, mcmc_samples_rllh]
            %      = obj.estimatePosterior(image, theta_init, confidence, num_samples, burnin, thin)
            %
            % Use MCMC sampling to sample from the posterior distribution and estimate the posterior
            % mean, a credible interval for upper and lower bounds on each parameter, and posterior
            % covariance. Optionally also returns the entire mcmc-posterior sample for further
            % post-processing, along with the RLLH at each sample.  Optional arguments are only computed
            % if required.
            %
            % MCMC sampling can be controlled with the optional num_samples, burn-in, and thin arguments.
            %
            % The confidence parameter sets the confidence-level for the credible interval bounds.  The
            % credible intervals bounds are per-parameter, i.e, each parameter at index i is individually
            % estimated to have a credible interval from lb(i) to ub(i), using the sample to integrate
            % out the other parameters.
            %
            % (in) image: a stack of n images to estimate
            % (in) theta_init: [optional] Initial theta guesses size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (in) confidence: [optional] desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
            % (in) num_samples: [optional] Number of (post-filtering) posterior samples to acquire. [default=obj.DefaultMCMCNumSamples]
            % (in) burnin: [optional] Number of samples to throw away (burn-in) on initialization [default=obj.DefaultMCMCBurnin]
            % (in) thin: [optional] Keep every # samples. Value of 0 indicates use the model default. This is suggested.
            %                       When thin=1 there is no thinning.  This is also a good option if rejections are rare. [default=obj.DefaultMCMCThin]
            % (out) posterior_mean: size:[NumParams,n] posterior mean for each image
            % (out) credible_lb: [optional] size:[NumParams,n] posterior credible interval lower bounds for each parameter for each image
            % (out) credible_ub: [optional] size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            % (out) posterior_cov: [optional] size:[NumParams,NumParams,n] posterior covariance matrices for each image
            % (out) mcmc_samples: [optional] size:[NumParams,max_samples,n] complete sequence of posterior samples generated by MCMC for each image
            % (out) mcmc_samples_rllh: [optional] size:[max_samples,n] relative log likelihood of sequence of posterior samples generated by MCMC. Each column corresponds to an image.

            image = obj.checkImage(image);
            nIms = size(image, obj.ImageDim+1);
            if nargin<3
                theta_init = [];
            end
            theta_init = obj.checkThetaInit(theta_init, nIms);
            if nargin<4
                confidence = obj.DefaultConfidenceLevel;
            elseif confidence<=0 || confidence>=1
                jerror('MappelBase:InvalidParameterValue', ['Bad confidence level for credible intervals: ', confidence]);
            end
            if nargin<5 || isempty(num_samples) || num_samples<=1
                num_samples = obj.DefaultMCMCNumSamples;
            end
            if nargin<6 || isempty(burnin) || burnin<0
                burnin = obj.DefaultMCMCBurnin;
            end
            if nargin<7
                thin = obj.DefaultMCMCThin;
            elseif thin < 0
                error('MappelBase:InvalidParameterValue', ['Bad MCMC thin value ', thin]);
            end
            [varargout{1:nargout}] = obj.call('estimatePosterior',image, theta_init, confidence, uint64(num_samples), uint64(burnin), uint64(thin));
        end

        function varargout = estimatePosteriorDebug(obj, image, theta_init, num_samples)
            % [sample, sample_rllh, candidates, candidate_rllh] = obj.estimatePosteriorDebug(image, theta_init, num_samples)
            %
            % Debugging routine.  Works on a single image.  Get out the exact MCMC sample sequence, as
            % well as the candidate sequence. Does not do burn-in or thinning.
            %
            % (in) image: a single image to estimate
            % (in) theta_init: [optional] Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (in) num_samples: [optional] Number of (post-filtering) posterior samples to acquire. [default=obj.DefaultMCMCNumSamples]
            % (out) sample: A size:[NumParams,num_samples] array of thetas samples
            % (out) sample_rllh: A size:[1,num_samples] array of relative log likelihoods at each sample theta
            % (out) candidates: [optional] size:[NumParams, num_samples] array of candidate thetas
            % (out) candidate_rllh: [optional] A size:[1, num_samples] array of relative log likelihoods at each candidate theta
            image=obj.checkImage(image);
            if nargin<3
                theta_init = [];
            end
            theta_init = obj.checkThetaInit(theta_init, 1);
            if nargin<4 || isempty(num_samples) || num_samples<=1
                num_samples = obj.DefaultMCMCNumSamples;
            end
             [varargout{1:nargout}] = obj.call('estimatePosteriorDebug', image, theta_init, num_samples);
        end

        %% Model property accessors
        function set.Hyperparams(obj, hyperparams)
            if numel(hyperparams) ~= obj.NumHyperparams
                error('MappelBase:setHyperParms:InvalidSize', ['Invalid size. Expecting: ' obj.NumHyperparams ' elements']);
            elseif any(hyperparams<0)
                error('MappelBase:setHyperparams:InvalidType', 'Prior should be non-negative');
            end
            if ~isempty(obj.Hyperparams) %avoid calling on fist setting in constructor
                obj.call('setHyperParameters',double(hyperparams)); %Set C++ object member variables
            end
            obj.Hyperparams = hyperparams; % Set local property
        end

        function set.HyperparamNames(obj,names)
            if numel(names) ~= obj.NumHyperparams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumHyperparams ' elements']);
            end
            try
                names = cellstr(names);
            catch Err
                switch ME.identifier
                    case 'MATLAB:cellstr:MustContainText'
                        error('MappelBase:InvalidType', 'Invalid types given.  Expected char arrays or string objects')
                    otherwise
                        rethrow(Err)
                end
            end
            if numel(unique(names)) ~= obj.NumHyperparams
                error('MappelBase:InvalidNames', 'Names should be unique');
            end
            if ~isempty(obj.HyperparamNames) %avoid calling on fist setting in constructor
                obj.call('setHyperparamNames',names); %Set C++ object member variables
            end
            obj.HyperparamNames = names; % Set local property
        end

        function set.ParamNames(obj,names)
            if numel(names) ~= obj.NumParams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumParams ' elements']);
            end
            try
                names = cellstr(names);
            catch Err
                switch ME.identifier
                    case 'MATLAB:cellstr:MustContainText'
                        error('MappelBase:InvalidType', 'Invalid types given.  Expected char arrays or string objects')
                    otherwise
                        rethrow(Err)
                end
            end
            if numel(unique(names)) ~= obj.NumParams
                error('MappelBase:InvalidNames', 'Names should be unique');
            end
            if ~isempty(obj.ParamNames) %avoid calling on fist setting in constructor
                obj.call('setParamNames',names); %Set C++ object member variables
            end
            obj.ParamNames = names; % Set local property
        end

        function set.ParamUnits(obj,units)
            if numel(units) ~= obj.NumParams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumParams ' elements']);
            end
            try
                units = cellstr(units);
            catch Err
                switch ME.identifier
                    case 'MATLAB:cellstr:MustContainText'
                        error('MappelBase:InvalidType', 'Invalid types given.  Expected char arrays or string objects')
                    otherwise
                        rethrow(Err)
                end
            end
            obj.ParamUnits = units; % Set local property
        end

        function set.ParamDescription(obj,desc)
            if numel(desc) ~= obj.NumParams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumParams ' elements']);
            end
            try
                desc = cellstr(desc);
            catch Err
                switch ME.identifier
                    case 'MATLAB:cellstr:MustContainText'
                        error('MappelBase:InvalidType', 'Invalid types given.  Expected char arrays or string objects')
                    otherwise
                        rethrow(Err)
                end
            end
            obj.ParamDescription = desc; % Set local property
        end

        function set.ParamUBound(obj,new_bound)
            if numel(new_bound) ~= obj.NumParams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumParams ' elements']);
            end
            if ~all(new_bound>obj.ParamLBound)
                if ~isempty(obj.ParamUBound) %avoid calling on fist setting in constructor
                    obj.call('setBounds',obj.ParamLBound, new_bound); %Set C++ object member variables
                end
                error('MappelBase:InvalidBound', 'Invalid bound must be greater than ParamLBound.');
            end    
            obj.ParamUBound = new_bound; % Set local property
        end

        function set.ParamLBound(obj,new_bound)
            if numel(new_bound) ~= obj.NumParams
                error('MappelBase:InvalidSize', ['Invalid size. Expecting ' obj.NumParams ' elements']);
            end    
            if ~isempty(obj.ParamLBound) %avoid calling on fist setting in constructor
                if ~all(new_bound<obj.ParamUBound)
                    error('MappelBase:InvalidBound', 'Invalid bound must be less than ParamUBound.');
                end
                obj.call('setBounds',new_bound, obj.ParamUBound); %Set C++ object member variables
            end
            obj.ParamLBound = new_bound; % Set local property
        end

        function [llh, theta_bg_mle] = uniformBackgroundModelLLH(obj, ims)
            % Test the model fit of a 1-parameter constant background model to the stack of images.
            % The MLE estimate for a 1-parameter background parameter is just the mean of the image.
            % The log-likelihood is calculated at this MLE estimate.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (out) llh: a length N vector of the LLH for each image for the constant-background model
            % (out) theta_bg_mle: a length N vector of the estimated MLE constant background.
            npixels = prod(obj.ImageSize);
            ims = reshape(ims,npixels,[]);
            theta_bg_mle = mean(ims)';
            llh =  log(theta_bg_mle).*sum(ims)' - npixels*theta_bg_mle - sum(gammaln(ims+1))';
        end

        function [pass, LLRstat] = modelComparisonUniform(obj, alpha, ims, theta_mle)
            % Do a LLH ratio test to compare the emitter model to a single parameter constant background model
            % The images are provided along with the estimated theta MLE values for the emitter model.
            % The LLH ratio test for nested models can be used and we compute the test statistice
            % LLRstat = -2*llh_const_bg_model + 2*llh_emitter_model.  This should be chisq distributed
            % with number of degrees of freedom given by obj.NumParams-1 since the const bg model has 1
            % param.
            % (in) alpha: 0<=alpha<1 - the certainty with which we should be sure to accept an emitter fit
            %                          vs. the uniform background model.  Values close to 1 reject more
            %                          fits.  Those close to 0 accept most fits.  At 0 only models where
            %                          the constant bg is more likely (even though it has only 1 free parameter)
            %                          will be rejected.  These arguably should always be rejected.  It
            %                          indicates an almost certainly bad fit.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (in) theta_mle: a double size:[NumParams, n] sequence of theta MLE values for each image in
            %                 ims
            % (out) pass: a boolean length N vector which is true if the emitter model passes for this test.
            %             in other words it is true for images with good fits that should be kept.  Images
            %             that fail could just as easily have been just random noise.
            % (out) LLRstat: -2*log(null_model_LH / emitter_model_LH);
            assert(0<=alpha && alpha<1);
            if nargin<4
                theta_mle = obj.estimate(ims);
            end
            llh_emitter_model = obj.LLH(ims,theta_mle);
            llh_const_bg_model = obj.uniformBackgroundModelLLH(ims);
            LLRstat = 2*(llh_emitter_model-llh_const_bg_model);
            threshold = chi2inv(alpha,obj.NumParams-1);
            pass = LLRstat > threshold;
        end

        function llh = noiseBackgroundModelLLH(obj, ims)
            % Test the model fit of an npixels-parameter all noise background model to the stack of images.
            % In this model each pixel has its own parameter and that pixels MLE will of course be the
            % value of the pixel itself.  Unlike the constant bg model there is no point to return the
            % MLE values themselves since they are just the images.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (out) llh: a length N vector of the LLH for each image for the constant-background model
            npixels = prod(obj.ImageSize);
            ims = reshape(ims,npixels,[]);
            llh = sum(ims.*log(ims)-ims-gammaln(ims+1))';
        end

        function [pass, bg_prob] = modelComparisonNoise(obj, alpha, ims, theta_mle)
            if nargin<4
                theta_mle = obj.estimate(ims);
            end
            emitter_aic = 2*obj.NumParams - 2*obj.LLH(ims,theta_mle);
            npixels = prod(obj.ImageSize);
            apparent_bg_aic = 2*npixels - 2*obj.noiseBackgroundModelLLH(ims);
            pass = emitter_aic < apparent_bg_aic;
            bg_prob = min(1,exp((emitter_aic - apparent_bg_aic)/2));
            pass = pass & (bg_prob<(1-alpha));
        end
        %% Evaluation and testing methods
    
        function [rmse, stddev, error] = evaluateEstimatorAt(obj, estimator, theta, nTrials, theta_init)
            %Evaluate this 2D estimator at a particular theta by running a fixed number of trials
            %
            % (in) estimator - Estimator name.  Can be any of the MAP estimator names or 'Posterior N' where N is a count
            % (in) theta - Parameter location to test at
            % (in) nTrials - Number of simulated images to fit
            % (out) rmse - [1,obj.NumParams]: The root-mean-squared error
            if nargin<5
                theta_init=[];
            end
            if nargin<4
                nTrials=1000;
            end
            theta = theta(:);
            ims = obj.simulateImage(theta,nTrials);
            
            if strncmpi(estimator,'posterior',9)
                count = str2double(estimator(10:end));
                confidence = 0.95;
                theta_est=obj.estimatePosterior(ims,theta_init,confidence,count);
            else
                theta_est=obj.estimate(ims,estimator, theta_init);
            end
            error = theta_est-repmat(theta,1,nTrials);
            rmse = sqrt(mean(error.*error,2));
            stddev = std(error,0,1);
        end

        function [theta_est,rmse,covariance] = evaluateEstimatorOn(obj, estimator, images, theta)
            %Evaluate this 2D estimator at a particular theta using the given samples which may have
            % been generated using different models or parameters.
            %
            % (in) estimator - String. Estimator name.  Can be any of the MAP estimator names or 'Posterior N' where N is a count
            % (in) images - size[flip(ImageSize),N] -  An array of sample images to test on
            % (in) theta - size:[NumParams] - true theta values for images.
            % (out) theta_est - size:[NumParams,N]: the estimated thetas
            % (out) rmse - size:[NumParams] root mean squared error at theta
            if strncmpi(estimator,'posterior',9)
                count = str2double(estimator(10:end));
                theta_init=[];
                confidence=0.95;
                theta_est = obj.estimatePosterior(images,theta_init,confidence,count);
            else
                theta_est = obj.estimate(images,estimator);
            end
            if nargin>3
                [~,Nims] = size(theta_est);
                rmse = sqrt(mean((theta_est-repmat(theta(:),1,Nims)).^2,2));
                covariance = cov((theta_est-repmat(theta(:),1,Nims))');
            end
        end
      
        function [theta_est_grid,est_var_grid]=mapEstimatorAccuracy(obj,estimator, sample_grid)
            % (in) estimator - String. Estimator name.  Can be any of the MAP estimator names or 'Posterior N' where N is a count
            % (in) sample_grid - size:[sizeY,sizeX,nTrials,gridsizeX,gridsizeY]
            %                     This should be produced by makeThetaGridSamples()
            % (out) theta_est_grid - size:[NumParams, nTrials, gridsizeX,gridsizeY] - estimated theta for each grid
            %                           image
            % (out) est_var_grid - size:[NumParams, nTrials, gridsizeX,gridsizeY] - estimated variance at each theta
            nTrials=size(sample_grid,3);
            gridsize=[size(sample_grid,4), size(sample_grid,5)];
            theta_est_grid=zeros([obj.NumParams, nTrials, gridsize]);
            est_var_grid=zeros([obj.NumParams, nTrials, gridsize]);
            h=waitbar(0,sprintf('Maping Accuracy Model:%s Estimator%s gridsize:%s',obj.Name, estimator, mat2str(gridsize)));
            for x=1:gridsize(1)
                for y=1:gridsize(2)
                    [theta_est, est_var]=obj.evaluateEstimatorOn(estimator, sample_grid(:,:,:,x,y));
                    theta_est_grid(:,:,x,y)=theta_est;
                    est_var_grid(:,:,x,y)=est_var;
                end
                waitbar(x/gridsize(1),h);
            end
            close(h);
        end

        function [theta_grid,sample_grid]=makeThetaGridSamples(obj, theta, gridsize, nTrials)
            % Make a grid of many theta values and corresponding sampled
            % images to test on for a particular theta.  This allows the
            % testing of the sensitivity of the estimator to the absolute
            % position in the image.  This generates the input for
            % mapEstimatorAccuacy().
            %
            % (in) theta - A theta to test over a spatial grid.  The x an y
            % are irrelevant as they will be modified for each grid point.
            % (in) gridsize - [X Y] The size of the grid of test locations.
            %                   suggest: [30 30]
            % (out) theta_grid - size:[obj.NumParams, nTrials,gridsize(1),gridsize(2)]
            % (out) sample_grid - size:[sizeY,sizeX,nTrials,gridsize(1),gridsize(2)]
            theta = theta(:);
            
            if isscalar(gridsize)
                gridsize=[gridsize gridsize];
            end
            theta_grid = zeros([obj.NumParams, nTrials, gridsize]);
            sample_grid = zeros([flip(obj.ImageSize), nTrials, gridsize]);
            grid_edges.x = linspace(0,double(obj.ImageSize(1)),gridsize(1)+1);
            grid_edges.y = linspace(0,double(obj.ImageSize(2)),gridsize(2)+1);
            for x=1:gridsize(1)
                for y=1:gridsize(2)
                    pixel_thetas=repmat(theta,1,nTrials);
                    e0=[grid_edges.x(x) grid_edges.y(y)]';
                    e1=[grid_edges.x(x+1) grid_edges.y(y+1)]';
                    %Make pixel thetas uniformly distributied over the pixel
                    pixel_thetas(1:2,:)=rand(2,nTrials).*repmat(e1-e0,1,nTrials)+repmat(e0,1,nTrials);
                    theta_grid(:,:,x,y)=pixel_thetas;
                    sample_grid(:,:,:,x,y)=obj.simulateImage(pixel_thetas);
                end
            end
        end
        
        %% Visualization methods
        
        function srim=superResolutionModel(obj, theta, theta_err, res_factor)
            % Generate a super-res image showing the estimated location and
            % error as a Gaussian
            % (in) theta - The parameter value to visualize
            % (in) theta_err - [optional] The RMSE error or sqrt(CRLB) that represents
            %                   the localization error. [default = run a simulation to estimate error] 
            % (in) res_factor - [optional] integer>1 - the factor to blow up the image
            %                   [default = 100]
            % (out) srim - double size:[sizeY*res_factor,sizeX*res_factor]
            %             A super-res rendering of the emitter fit position.
            if nargin<4
                res_factor=100;
            end
            if nargin<3
                theta_err = obj.evaluateEstimatorAt('Newton',theta);
            end
            srimsize=obj.ImageSize*res_factor;
            theta(1)=theta(1)*res_factor;
            theta(2)=theta(2)*res_factor;
            theta_err(1)=theta_err(1)*res_factor;
            theta_err(2)=theta_err(2)*res_factor;

            xs=0:srimsize(1)-1;
            X=0.5*(erf(((xs+1)-theta(1))/(sqrt(2)*theta_err(1)))-erf((xs-theta(1))/(sqrt(2)*theta_err(1))));
            ys=0:srimsize(2)-1;
            Y=0.5*(erf(((ys+1)-theta(2))/(sqrt(2)*theta_err(2)))-erf((ys-theta(2))/(sqrt(2)*theta_err(2))));
            srim=theta(3)*Y'*X;           
        end
        
        function plotAccuracyMap(obj, grid)
            x=[0.5,obj.ImageSize(1)-0.5];
            y=[0.5,obj.ImageSize(2)-0.5];
            imagesc(x,y,grid');
            xlabel('x (pixels)');
            ylabel('y (pixels)');
        end

    end% public methods

    methods (Access = protected)
        function [theta, llh, obsI, stats]=estimate_GPUGaussMLE(obj, image)
            if ~ispc()
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE on this archetecture');
            end
            if obj.psf_sigma(1)~=obj.psf_sigma(2) || obj.ImageSize(1)~=obj.ImageSize(2)
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE as boxsize or psf_sigma is not uniform');            
            end
            if obj.GPUGaussMLEFitType<1
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE on this model type');
            end
            data=single(image);
            psf=obj.psf_sigma(1);
            iters=obj.GPUGaussMLE_Iterations;
            type=obj.GPUGaussMLEFitType;
            [P, ~, LL]=gpugaussmlev2(data, psf, iters, type);
            theta=double(P');
            theta([1,2],:) = theta([2,1],:); %Swap dims
            theta([1,2],:) = theta([1,2],:)+0.5; %Correct for 1/2 pixel
            obsI = obj.observedInformation(image, theta);
            llh = LL;
            stats.iterations=iters;
            stats.fittype=type;
        end
        
        function [theta, llh, obsI, stats, sequence, sequence_llh]=estimateDebug_GPUGaussMLE(obj, image)
            if ~ispc()
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE on this archetecture');
            end
            if obj.psf_sigma(1)~=obj.psf_sigma(2) || obj.ImageSize(1)~=obj.ImageSize(2)
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE as boxsize or psf_sigma is not uniform');            
            end
            if obj.GPUGaussMLEFitType<1
                error('MappelBase:estimateGPUGaussMLE','Unable to run GPUGaussMLE on this model type');
            end
            data=single(image);
            psf=obj.psf_sigma(1);
            iters=obj.GPUGaussMLE_Iterations;
            type=obj.GPUGaussMLEFitType;
            [P, ~, LL]=gpugaussmlev2(data, psf, iters, type);
            theta=double(P');
            theta([1,2],:) = theta([2,1],:); %Swap dims
            theta([1,2],:) = theta([1,2],:)+0.5; %Correct for 1/2 pixel
            obsI = obj.observedInformation(image, theta);
            llh = LL;
            stats.iterations=iters;
            stats.fittype=type;
            sequence=theta;
            sequence_llh=LL;
        end

        function [theta, llh, obsI, stats]=estimate_fminsearch(obj, image, theta_init)
            %
            % Uses matlab's fminsearch (Simplex Algorithm) to maximize the LLH for a stack of images.
            % This is available in the core Matlab and does not require the optimization toolbox
            %
            % Uses LLH function evaluation calculations from C++ interface.
            % (in) image - a stack of N double images size:[flip(ImageSize), n]
            % (in) theta_init: [optional] Initial theta guesses size (NumParams x n).  Values of 0 indicate
            %            that we have no initial guess and the estimator should form its own guess.
            % (out) theta: size:[NumParams, N] double of estimated theta values
            % (out) llh: [optional] a size:[1, N] double of the log likelihood at each theta estimate.
            % (out) obsI: [optional] size:[NumParams, NumParams, N] estimate of the CRLB for each parameter estimate.
            %             This gives the approximate variance in the theta parameters
            % (out) stats: [optional] A 1x1 struct of fitting statistics.
            N = size(image,obj.ImageDim+1);
            if nargin<3 || isempty(theta_init) || ~all(obj.thetaInBounds(theta_init))
                theta_init = obj.estimate(image,'Heuristic');
            elseif isvector(theta_init)
                theta_init = repmat(theta_init',1,N);
            end
            [theta, llh, obsI, stats] = obj.estimate_fminsearch_core(image, theta_init);
        end

        function [theta, llh, obsI, sequence, sequence_llh,stats] = estimateDebug_fminsearch(obj, image, theta_init)
            max_iter = obj.DefaultMatlabFminsearchMaxIter;
            sequence = zeros(obj.NumParams, max_iter);
            nseq = 0;
            function stop = output(theta, ~, state)
                nseq = nseq+1;
                sequence(:,nseq)=theta;
                stop = strcmp(state,'done');
            end
            if nargin<3 || isempty(theta_init) || ~obj.thetaInBounds(theta_init)
                theta_init = obj.estimate(image,'Heuristic');
            end
            opts.OutputFcn = @output;
            opts.MaxIter=max_iter;
            [theta, llh, obsI, stats] = obj.estimate_fminsearch_core(image, theta_init, opts);
            sequence = sequence(:,1:nseq);
            in_bounds = find(obj.thetaInBounds(sequence));
            sequence_llh=-inf(nseq,1);
            rllh = obj.modelRLLH(image, sequence(:,in_bounds));
            sequence_llh(in_bounds)=rllh;
            stats.sequenceLen = size(sequence,2);
        end
        
        function [theta, llh, obsI, stats] = estimate_fminsearch_core(obj, image, theta_init, extra_opts)
            Nims = size(theta_init,2);
            Np = obj.NumParams;
            
            opts = optimset('fminsearch');
            opts.MaxIter = obj.DefaultMatlabFminsearchMaxIter;
            opts.Diagnostics = 'off';
            opts.Display='off';
            opts.TolX = obj.DefaultMatlabOptimizerTolerance;
            opts.TolFun = obj.DefaultMatlabOptimizerTolerance;

            if nargin==4
                for name = fieldnames(extra_opts) 
                    opts.(name{1}) = extra_opts.(name{1});
                end
            end
            problem.solver = 'fminsearch';
            problem.options = opts;
            function val = callback(im,x)
                if any(x-obj.ParamLBound<1e-6) || any(obj.ParamUBound-x<1e-6)
                    val = inf;
                else
                    val = -obj.modelLLH(im,x);
                end
            end
            llh=zeros(Nims,1);
            obsI=zeros(Np,Np,Nims);
            theta=zeros(Np,Nims);
            stats.flags = zeros(Nims,1);
            
            stats.num_exit_error = 0;
            stats.num_exit_success = 0;
            stats.num_exit_max_iter = 0;
            stats.total_iterations = 0;
            stats.total_backtracks = 0;
            stats.total_fun_evals = 0;
            stats.total_der_evals = 0;
            stats.algorithm = 'fminsearch';
            stats.num_estimations = Nims;
            stats.TolX = opts.TolX;
            stats.TolFun = opts.TolFun;
            stats.MaxIter = opts.MaxIter;
            for n=1:Nims
                im=image(:,:,n);
                problem.objective = @(x) callback(im,x);
                problem.x0 = theta_init(:,n);
                [theta(:,n), fval, stats.flags(n), out] = fminsearch(problem);
                obsI(:,:,n) = obj.observedInformation(im, theta(:,n));
                llh(n) = -fval;
                
                switch stats.flags(n)
                    case 1
                        stats.num_exit_success = stats.num_exit_success+1;
                    case 0
                        stats.num_exit_max_iter = stats.num_exit_max_iter+1;
                    case -1
                        stats.num_exit_error = stats.num_exit_error+1;
                end
                stats.total_iterations = stats.total_iterations + out.iterations;
                stats.total_fun_evals = stats.total_fun_evals + out.funcCount;
                stats.total_der_evals = stats.total_der_evals + 1;
            end
            
        end
        

        function [theta, llh, obsI, stats]=estimate_toolbox(obj, image, theta_init, algorithm)
            %
            % Requires: matlab optimization toolbox
            %
            % Uses matlab's fminunc or fmincon to maximize the LLH for a stack of images.  
            % Uses LLH grad and hessian calculations from C++ interface.
            % Note that 'interior-point' and 'trust-region-reflective' use fmincon and use simple bounded 
            % constraints on the parameters.
            % 'quasi-newton', and 'trust-region' algorithms use fminunc and perform unconstrained
            % optimizations.  
            % All methods use the full hessian except for 'quasi-newton'.  All methods also use the full
            % gradient values.
            %
            % (in) image - A stack of N images. type: double size:[flip(ImageSize), N]
            % (in) theta_init - [optional] size:[NumParams, N] array giving initial theta value for each image
            %                   Default: Use Heuristic.
            % (in) algorithm - [optional] string: The algorithm to choose.
            %                    ['quasi-newton', 'interior-point', 'trust-region', 'trust-region-reflective']
            %                    [default: 'trust-region']
            % (out) theta - size:[NumParams, N]. Optimal theta value for each image
            % (out) llh - size:[NumParams, N]. LLH value at optimal theta for each image
            % (out) obsI - size:[NumParams, N]. CRLB value at each theta value.
            % (out) stats - statistics of fitting algorithm's performance.
            if nargin<4
                algorithm = 'trust-region';
            end
            N = size(image,obj.ImageDim+1);
            if nargin<3 || isempty(theta_init) || ~all(obj.thetaInBounds(theta_init))
                theta_init = obj.estimate(image,'Heuristic');
            elseif isvector(theta_init)
                theta_init = repmat(theta_init',1,N);
            end
            [theta, llh, obsI, stats] = obj.estimate_toolbox_core(image, theta_init, algorithm);
        end
            
        function [theta, llh, obsI, sequence, sequence_llh, stats] = estimateDebug_toolbox(obj, image, theta_init, algorithm)
            %
            % Requires: matlab optimization toolbox
            %
            % Uses matlab's fminunc or fmincon to maximize the LLH for a stack of images.  
            % Uses LLH grad and hessian calculations from C++ interface.
            % Note that 'interior-point' and 'trust-region-reflective' use fmincon and use simple bounded 
            % constraints on the parameters.
            % 'quasi-newton', and 'trust-region' algorithms use fminunc and perform unconstrained
            % optimizations.  
            % All methods use the full hessian except for 'quasi-newton'.  All methods also use the full
            % gradient values.
            %
            % (in) image - A single image. type: double size:[flip(ImageSize)]
            % (in) theta_init - [optional] size:[NumParams, 1] initial theta value for image
            %                   Default: Use Heuristic.
            % (in) algorithm - [optional] string: The algorithm to choose.
            %                    ['quasi-newton', 'interior-point', 'trust-region', 'trust-region-reflective']
            %                    [default: 'trust-region']
            % (out) theta - size:[NumParams, 1]. Optimal theta value for image
            % (out) llh - size:[NumParams, 1]. LLH value at optimal theta for image
            % (out) obsI - size:[NumParams, NumParams].  Observed information at theta.
            % (out) stats - statistics of fitting algorithm's performance.
            % (out) sequece - size[NumParams, K]. For K steps, return each intermediate theta value we
            %                                   evaluated the objective function at.
            % (out) sequece_llh - size[1, K]. For K steps, return each intermediate theta llh value we
            %                                 evaluated.
            if nargin<4
                algorithm = 'trust-region';
            end
            max_iter=obj.DefaultMatlabToolboxMaxIter;
            sequence=zeros(obj.NumParams, max_iter);
            nseq=0;
            function stop = output(theta, ~, state)
                nseq = nseq+1;
                sequence(:,nseq)=theta;
                stop = strcmp(state,'done');
            end
            if nargin<3 || isempty(theta_init) || ~obj.thetaInBounds(theta_init)
                theta_init = obj.estimate(image,'Heuristic');
            end
            opts.OutputFcn = @output;
            opts.MaxIter=max_iter;
            [theta, llh, obsI, stats] = obj.estimate_toolbox_core(image, theta_init, algorithm, opts);
            sequence = sequence(:,1:nseq);
            in_bounds = find(obj.thetaInBounds(sequence));
            sequence_llh=-inf(nseq,1);
            rllh = obj.modelRLLH(image, sequence(:,in_bounds));
            sequence_llh(in_bounds)=rllh;
            stats.sequenceLen = size(sequence,2);
        end
        
        
        function [theta, llh, obsI, stats] = estimate_toolbox_core(obj, image, theta_init, algorithm, extra_opts)
            switch algorithm
                case {'quasi','quasi-newton','quasi-newton'}
                    algorithm = 'quasi-newton';
                    solver = @fminunc;
                case {'trust','trust-region','trustregion'}
                    algorithm = 'trust-region';
                    solver = @fminunc;
                case {'interior', 'interior-point', 'interiorpoint'}
                    algorithm = 'interior-point';
                    solver = @fmincon;
                case {'reflective','trust-region-reflective','trustregionreflective'}
                    algorithm = 'trust-region-reflective';
                    solver = @fmincon;
                case {'sqp'}
                    algorithm = 'sqp';
                    solver = @fmincon;
                case {'active-set','activeset'}
                    algorithm = 'active-set';
                    solver = @fmincon;
                otherwise
                    error('MappelBase:estimateDebug_fmincon','Unknown maximization method: "%s"', algorithm);
            end
            problem.solver = func2str(solver);
            opts = optimoptions(problem.solver);
            opts.Algorithm = algorithm;
            opts.SpecifyObjectiveGradient = true;
            opts.Diagnostics = 'off';
            %opts.Display = 'iter-detailed';
%             opts.Diagnostics = 'off';
            opts.Display = 'off';
            opts.MaxIterations = obj.DefaultMatlabToolboxMaxIter;
            opts.FunctionTolerance = obj.DefaultMatlabOptimizerTolerance;
            opts.StepTolerance = obj.DefaultMatlabOptimizerTolerance;
            opts.OptimalityTolerance = obj.DefaultMatlabOptimizerTolerance;
            function varargout = objective(im,theta)
                if ~obj.thetaInBounds(theta)
                    varargout{1}=inf;
                    if nargout>1
                        varargout{2}=zeros(numel(theta),1);
                        if nargout>2
                            varargout{3}=zeros(numel(theta));
                        end
                    end
                    return
                end
                [varargout{1:nargout}] = obj.modelObjective(im,theta,1);
            end
            function hess = hessFunc(im,theta)
                if ~obj.thetaInBounds(theta)
                    hess = zeros(numel(theta));
                else
                    hess = -obj.modelHessian(im,theta);
                end
            end
            %Algorithm sepecifc opts
            switch algorithm
                case 'quasi-newton'
                    %Cannot supply hessian for quasi-newton
                    opts.HessUpdate = 'bfgs'; %Method to choose search direction. ['bfgs', 'steepdesc', 'dfp']                    
                case 'trust-region'                   
                    opts.HessianFcn = 'objective'; %Third arg in objective function.
                    opts.SubproblemAlgorithm = 'factorization'; % more accurate step
                case 'trust-region-reflective'                   
                    opts.HessianFcn = 'objective'; %Third arg in objective function.
                    opts.SubproblemAlgorithm = 'factorization'; % more accurate step
                case 'interior-point'
                    opts.SubproblemAlgorithm = 'factorization'; % more accurate step
            end
            % Merge in extra_opts
            if nargin>=5
                for name = fieldnames(extra_opts) 
                    opts.(name{1}) = extra_opts.(name{1});
                end
            end
            
            problem.options = opts;
            
            Nims = size(image, obj.ImageDim+1); %number of images to process
            theta = zeros(obj.NumParams, Nims);
            llh = zeros(Nims,1);
            flags = zeros(Nims,1);
            obsI = zeros(obj.NumParams,obj.NumParams,Nims);
            
            %Prepare stats
            stats.flags = zeros(Nims,1);
            stats.step_norms = zeros(Nims,1);
            stats.firstorderopt = zeros(Nims,1);
            
            stats.num_exit_error = 0;
            stats.num_exit_grad_ratio = 0;
            stats.num_exit_step_size = 0;
            stats.num_exit_function_value = 0;
            stats.num_exit_model_improvement = 0;
            stats.num_exit_max_iter = 0;
            stats.total_iterations = 0;
            stats.total_backtracks = 0;
            stats.total_fun_evals = 0;
            stats.total_der_evals = 0;
            stats.FunctionTolerance = opts.FunctionTolerance;
            stats.StepTolerance = opts.StepTolerance;
            stats.OptimalityTolerance = opts.OptimalityTolerance;
            switch algorithm
                case 'trust-region'
                    stats.total_cgiterations = 0;
            end
            
            tic();
            for n=1:Nims
                im = image(:,:,n);
                problem.x0 = theta_init(:,n);
                problem.objective = @(theta,~) objective(im,theta);
                switch algorithm
                    case 'interior-point'
                        problem.options.HessianFcn = @(t,~) hessFunc(im,t);
                end
                [theta(:,n), fval, stats.flags(n), out] = solver(problem);
                llh(n)=-fval;
                obsI(:,:,n) = obj.observedInformation(im,theta(:,n));
                switch stats.flags(n)
                    case 5
                        stats.num_exit_model_improvement = stats.num_exit_model_improvement+1;
                    case 3
                        stats.num_exit_function_value = stats.num_exit_function_value+1;
                    case 2
                        stats.num_exit_step_size = stats.num_exit_step_size+1;
                    case 1
                        stats.num_exit_grad_ratio = stats.num_exit_grad_ratio+1;
                    case 0
                        stats.num_exit_max_iter = stats.num_exit_max_iter+1;
                    case {-1,-3}
                        stats.num_exit_error = stats.num_exit_error+1;
                end
                stats.step_norms(n) = norm(out.stepsize);
                stats.firstorderopt(n) = out.firstorderopt;
                stats.total_iterations = stats.total_iterations + out.iterations;
                stats.total_fun_evals = stats.total_fun_evals + out.funcCount;
                stats.total_der_evals = stats.total_der_evals + out.funcCount;
                switch algorithm
                    case 'trust-region'
                        stats.total_cgiterations = stats.total_cgiterations + out.cgiterations;
                end
            end
            stats.total_walltime = toc();
            stats.mean_walltime = stats.total_walltime/Nims;
            stats.algorithm = algorithm;
            stats.flags = flags;
            stats.num_estimations = Nims;
            stats.num_threads=1;            
        end

%         function varargout = errorBoundsProfileLikelihood_fzero_debug(obj, image, estimate_parameter, theta_mle, theta_mle_rllh, obsI, llh_delta)
%             % [profile_lb, profile_ub, seq_lb, seq_ub, seq_lb_rllh, seq_ub_rllh, stats]
%             %    = obj.errorBoundsProfileLikelihood_fzero_debug(image, estimate_parameter, theta_mle, theta_mle_rllh, obsI, llh_delta)
%             %
%             % [DEBUGGING]
%             % Compute the profile log-likelihood bounds for a single images , estimating upper and lower bounds for each requested  parameter.
%             % Uses the Venzon and Moolgavkar (1988) algorithm.
%             %
%             % (in) image: a single images
%             % (in) theta_mle:  theta ML estimate
%             % (in) theta_mle_rllh: relative-log-likelihood at image.
%             % (in) obsI: a observed fisher information matrix at theta_mle
%             % (in) estimate_parameter: integer index of parameter to estimate size:[NumParams]
%             % (in) llh_delta: [optional] Negative number, indicating LLH change from maximum at the profile likelihood boundaries.
%             %                  [default: confidence=0.95; llh_delta = -chi2inv(confidence,1)/2;]
%             % (out) profile_lb:  scalar lower bound for parameter
%             % (out) profile_ub:  scalar upper bound for parameter
%             % (out) seq_lb: size:[NumParams,Nseq_lb]  Sequence of Nseq_lb points resulting from VM algorithm for lower bound estimate
%             % (out) seq_ub: size:[NumParams,Nseq_ub]  Sequence of Nseq_ub points resulting from VM algorithm for upper bound estimate
%             % (out) seq_lb_rllh: size:[Nseq_lb]  Sequence of RLLH at each of the seq_lb points
%             % (out) seq_ub_rllh: size:[Nseq_ub]  Sequence of RLLH at each of the seq_ub points
%             % (out) stats: struct of fitting statistics.
%             image = obj.checkImage(image);
%             if nargin<3
%                 estimate_parameter = 1;
%             end
%             if nargin<4
%                 [theta_mle, theta_mle_rllh, obsI] = obj.estimate(image);
%             else
%                 theta_mle = obj.checkTheta(theta_mle);
%             end
%             if nargin<7
%                 llh_delta = -chi2inv(obj.DefaultConfidenceLevel,1)/2;
%             end
%             [varargout{1:nargout}] = obj.call('errorBoundsProfileLikelihoodDebug',image, theta_mle, theta_mle_rllh, obsI, uint64(estimate_parameter)-1, llh_delta);
%         end
%         
%     function v=profile_delta(x,fixed_p)
%         fixed_parameters=zeros(1,obj.NumParams);
%         fixed_parameters(fixed_p)=1;
%         prllh = obj.estimateProfileLikelihood(im,fixed_parameters,x,'TrustRegion',theta_mle);
%         v=prllh-(mle_rllh+c);
%     end
%     prof_lb=zeros(obj.NumParams,1);
%     prof_ub=zeros(obj.NumParams,1);
%     for i=1:obj.NumParams
%         epsilon=1e-8;
%         if(profile_delta(obj.ParamLBound(i)+epsilon,i)>0)
%             prof_lb(i) = obj.ParamLBound(i)+epsilon;
%         else
%             prof_lb(i) = fzero(@(x) profile_delta(x,i),[obj.ParamLBound(i)+epsilon,theta_mle(i)]); 
%         end
%         if isfinite(obj.ParamUBound(i))
%             if(profile_delta(obj.ParamUBound(i)-epsilon,i)>0)
%                 prof_ub(i) = obj.ParamUBound(i)-epsilon;
%             else
%                 prof_ub(i) = fzero(@(x) profile_delta(x,i),[theta_mle(i),obj.ParamUBound(i)-epsilon]); 
%             end
%         else
%             prof_ub(i) = fzero(@(x) profile_delta(x,i),[theta_mle(i),theta_mle(i)*20]); 
%         end
%     end
        

    end % protected methods
    
    methods (Access=public)
        function definiteM = choleskyMakePositiveDefinite(obj,M)
            definiteM = obj.callstatic('positiveDefiniteCholeskyApprox',M);
        end
    end
    
    methods (Static = true)
        function [C,D]=cholesky(A)
            [valid, C, d] = MappelBase.callstatic('cholesky',A);
            if(~valid)
                error('MappelBase:cholesky','Matrix not positive definite symmetric');
            end
            D = diag(d);
        end

        function [C,D, is_positive]=modifiedCholesky(A)
            [is_positive, C, d] = MappelBase.callstatic('modifiedCholesky',A);
            D = diag(d);
        end

        function [x, modified]=choleskySolve(A,b)
            [modified, x] = MappelBase.callstatic('choleskySolve',A,b);
        end

        function definiteM = choleskyMakeNegativeDefinite(M)
            definiteM = MappelBase.callstatic('negativeDefiniteCholeskyApprox',M);
        end

        function fig = viewDipImage(image, fig)
            if nargin==1
                fig=figure();
            end
            if ~isa(image,'dip_image')
                image = dip_image(image);
            end
            dipshow(fig,image);
            diptruesize(100*1000./max(size(image)))
            dipmapping(fig,'colormap',hot);
        end
    end %public static methods
    
    methods (Access=protected)        
        function image=checkImage(obj, image)
            ndim=length(obj.ImageSize);
            if (ndims(image) < ndim) || (ndims(image) > ndim+1)
                error('MappelBase:checkImage', 'Invalid image dimension');
            end
            if ndim==2
                if size(image,1)~=obj.ImageSize(2) || size(image,2)~=obj.ImageSize(1) 
                    error('MappelBase:checkImage', 'Invalid image shape');
                end
            else
                imsz=size(image);
                if any(imsz(ndim:-1:1)~=obj.ImageSize)
                    error('MappelBase:checkImage', 'Invalid image shape');
                end
            end
            if ~all(image>=0)
                error('MappelBase:checkImage', 'Invalid image');
            end
            image=double(image);
        end
        
         function count=checkCount(~, count)
            if ~isscalar(count) || count<=0
                error('MappelBase:checkCount', 'Invalid count');
            end
            count=int32(count);
        end

        function theta = checkTheta(obj, in_theta)
            %Checks that in_theta is in correct shape [NumParams, N].
            %(in) in_theta - [NumParams,N] vector given as input by user
            %(out) theta - [NumParams,N] corrected theta.
            if size(in_theta,1) ~= obj.NumParams
                if length(in_theta) == obj.NumParams
                    theta = double(in_theta');
                else
                    error('MappelBase:checkTheta', 'Invalid theta shape. Expected %i params',obj.NumParams);
                end
            else
                theta = double(in_theta);
            end
        end
        
        function theta_init = checkThetaInit(obj, theta_init, nIms)
            % Ensure theta_init is size:[NumParams,nIms]
            if isempty(theta_init)
                theta_init = zeros(obj.NumParams,nIms);
            elseif isvector(theta_init)
                if numel(theta_init) ~= obj.NumParams
                    error('Mappel:thetaInitValue','Invalid theta init shape');
                end
                theta_init = repmat(theta_init(:),1,nIms);
            elseif any(size(theta_init) ~= [obj.NumParams, nIms])
                error('MappelBase:InvalidValue','Invalid theta init shape' );
            end    
        end
        
        function mask = paramMask(obj, names)
            mask = zeros(obj.NumParams,1);
            for n = 1:obj.NumParams
                param = obj.ParamNames{n};
                mask(n) = any(cellfun(@logical, cellfun(@(s) strcmp(s,param), names, 'Uniform', 0)));
            end
        end
    end % protected methods
end %classdef
