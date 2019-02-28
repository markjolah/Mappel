
classdef MappelBase < MexIFace.MexIFaceMixin
    % MappelBase.m
    %
    % Mappel base class interface for all point-emitter localization models.
    %
    % This base class implements most of the methods for each of the Mappel Models classes.  
    %
    % 1D Models
    %
    %
    % Mappel.MappelBase Properties
    %    ImageSize -  1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
    %    PSFSigmaMin - Minimum gaussian point-spread function sigma size in pixels 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
    %    PSFSigmaMax - Minimum gaussian point-spread function sigma size in pixels 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
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
    %  * Mappel.MappelBase.samplePrior - Sample typical parameter (theta) values from the prior using Hyperparams
    %  * Mappel.MappelBase.simulateImage - Simulate image with noise given one or more parameter (theta) values.
    %  * Mappel.MappelBase.estimate - Estimate the emitter parameters from a stack of images using maximum-likelihood. 
    %  * Mappel.MappelBase.estimate - Point estimates of the model parameters from a stack of images [MLE/MAP]. 
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
                           'CGauss',...             %[C(OpenMP)] C Diagonal NewtonRaphson implementation from Smith et. al. Nat Methods (2010)
                           'CGaussHeuristic',...    %[C(OpenMP)] C Heuristic starting guesstimate implementation from Smith et. al. Nat Methods (2010)
                           'GPUGauss',...           %[C(CUDA)] CUDA implementation from Smith et.al. Nat Methods (2010)
                           'matlab-fminsearch',...              %[MATLAB (fminsearch)] Single threaded fminsearch Nelder-Mead Simplex (derivative free) optimization from Matlab core
                           'matlab-quasi-newton',...            %[MATLAB (fminunc)] Single threaded Quasi-newton fminunc optimization from Matlab Optimization toolbox
                           'matlab-trust-region',...            %[MATLAB (fminunc)] Single threaded trust-region fminunc optimization from Matlab Optimization toolbox
                           'matlab-trust-region-reflective',... %[MATLAB (fmincon)] Single threaded trust-region optimization from Matlab Optimization toolbox
                           'matlab-interior-point'...           %[MATLAB (fmincon)] Single threaded interior-point optimization from Matlab Optimization toolbox
                           };
    end
    properties (Abstract=true, Constant=true)
        ImageDim; %Dimensionality of images
    end
    properties (SetAccess = protected)
        NumParams; %Number of model params (i.e., model dimensionality)
        NumHyperparams; %Number of hyper-parameters, (i.e., the parameters to the model's prior)
    end %read-only properties

    % These properties use get and set methods to read and modify C++ object member variables.
    % Size and type are checked, and cannot be changed.
    properties
        ImageSize; % 1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        PSFSigmaMin; %1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        PSFSigmaMax; %1D:[X], 2D:[X Y], or Hyperspectral:[X Y L]
        Hyperparams; % [NumHyperparams,1] vector of hyperparams.
        ParamNames;  % [NumParams,1] cellarray of param names
        HyperparamNames; % [NumHyperparams,1] cellarray of hyperparam names
        ParamUBound; %Upper bound for each parameter (dimension) inf for unbounded above.  Controls bounded optimization/estimation in C++.
        ParamLBound; %Lower bound for each parameter (dimension) -inf for unbounded below. Controls bounded optimization/estimation in C++.
    end % accessor properties to MexIFace

    %These properties have no C++ backing, but can be set to control the plotting and presentation of data
    properties
        ParamUnits; % [NumParams,1] cellarray of param unit types names (as char strings)
        ParamDescription;  % [NumParams,1] cellarray of param descriptions (as char strings)
    end

    %Default model settings
    properties 
        DefaultEstimatorMethod = 'TrustRegion'; %Set this to control the default optimization method for MLE/MAP estimation
        DefaultMCMCNumSamples = 300; % Number of final samples to use in estimation of posterior properties (mean, credible interval, cov, etc.)
        DefaultMCMCBurnin = 10; % Number of samples to throw away (burn-in) on initialization
        DefaultMCMCThin = 0; % Keep every # samples. [Value of 0 indicates use the model default. This is suggested.]
        DefaultConfidenceLevel = 0.95; % Default level at which to estimate confidence intervals must be in range (0,1).
        DefauktGPUGaussMLE_Iterations = 15; %GPUGaussMLE is supported for Win64 only.
    end

    properties (Access=protected)
        GPUGaussMLEFitType;
    end
    
    methods
        function obj = MappelBase(iface, imsize, psf_sigma_min, psf_sigma_max)
            % fixed-sigma models: obj = MappelBase(iface, imsize, psf_sigma)
            %  free-sigma models: obj = MappelBase(iface, imsize, psf_sigma_min, psf_sigma_max)
            %
            % (in) iface: The iface object mex object nemae
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
                psf_sigma_max = double(psf_sigma_max(:)');
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
            % Convenience method to get a hyperparameter value by name
            %
            % (in) name: name of hyperparam (case insensitive)
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
            % Convenience method to set a hyperparameter value by name
            %
            % (in) name: name of hyperparam (case insensitive)
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
            % May also bound theta away from edge by epsilon.
            %
            % (in) theta - A theta to correct to ensure it is in bounds
            % (out) bounded_theta - A corrected theta that is now in-bounds
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
            % (in) theta - A theta to check if it is valid
            % (out) ibounds - logical: True if the theta is valid
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
            % then n images are simulated, each with a seperate theta, and count is ignored.
            % The noise model of the Model class is used to add noise to the sampled model images.
            %
            % (in) theta: size:[NumParams, 1] or [NumParams, n] double theta value.
            % (in) count: (optional) the number of independant images to generate. Only used if theta is a single parameter.
            % (out) image: a stack of n images, all sampled with params theta.
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
            % (in) count: (optional) the number of independent images to generate. 
            %             Only used if theta is a single parameter.
            % (out) image: a dim_images object holding a stack of n images all sampled with params theta
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
            % (out) llh: size:[1,max(M,N)] double of log_likelihoods
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
            % grad = obj.modelGrad(image, theta) - Compute the model gradiant.
            %
            % This takes in a N images and M thetas.  If M=N=1, then we return a single Grad.  If there
            % are N=1 images and M>1 thetas, we return M Grads of the same image with each of the thetas.
            % Otherwise, if there is M=1 thetas and N>1 images, then we return N Grads for each of the
            % images given theta
            %
            % (in) image: An image stack of N images.  For 2D images this is size:[SizeY, SizeX, N]
            % (in) theta: A size:[NumParams, M] stack of theta values
            % (out) grad: a (NumParams X max(M,N)) double of gradiant vectors
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
            % (out) hess: a (NumParams X NumParams X max(M,N)) double of hessian matricies
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
            % (out) hess: a (NumParams X NumParams X max(M,N)) double of hessian matricies
            definite_hess = obj.modelHessian(image,theta);
            definite_hess = obj.callstatic('negativeDefiniteCholeskyApprox', definite_hess);
        end

        function varargout = modelObjective(obj, image, theta, negate)
            % [rllh,grad,hess,definite_hess,llh] = obj.modelObjective(image, theta, negate)
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
            % (out) grad: (optional) grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: (optional) negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
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
            % (in) negate: (optional) boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh: relative log likelihood scalar double
            % (out) grad: (optional) grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: (optional)  negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
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
            % (in) negate: (optional) boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh: relative log likelihood scalar double
            % (out) grad: (optional) grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: (optional) negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
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
            % (in) negate: (optional) boolean. true if objective should be negated, as is the case with
            %                 matlab minimization routines
            % (out) rllh:  relative log likelihood scalar double
            % (out) grad: (optional) grad of log likelihood scalar double size:[NumParams,1]
            % (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams]
            % (out) definite_hess: (optional) negative(positive)-definite hessian of log likelihood double size:[NumParams,NumParams]
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
            % (out) grad: (optional) components of grad of log likelihood size:[NumParams,NumComponents*]  
            %             * there is only a single extra grad component added for prior in MAP models.
            % (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams,NumComponents*] 
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
            % This is equivalent to the negative hessian of the llh.
            %
            % (in) image: a single image or a stack of images
            % (in) theta_mle: an estimated  MLE theta size:[NumParams,N].  One theta_mle per image.
            % (out) obsI: the observed information matrix size:[NumParams, NumParams]
            image = obj.checkImage(image);
            theta_mle = obj.checkTheta(theta_mle);
            obsI = -obj.call('modelHessian',image, theta_mle);
        end

        function [Observed_lb, observed_ub] = errorBoundsObserved(obj, image, theta_mle, confidence, obsI)
            % [observed_lb, observed_ub] = obj.errorBoundsObserved(image, theta_mle, confidence, obsI)
            %
            % Compute the error bounds using the observed information at the MLE estimate theta_mle.
            %
            % (in) image: a single image or a stack of n images
            % (in) theta_mle: the MLE estimated theta size:[NumParams,n]
            % (in) confidence: (optional) desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (in) obsI: (optional) observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n]
            % (out) observed_lb: the observed-information-based confidence interval lower bound for parameters, size:[NumParams,n]
            % (out) observed_ub: the observed-information-based confidence interval upper bound for parameters, size:[NumParams,n]
            image = obj.checkImage(image);
            theta_mle = obj.checkTheta(theta_mle);
            if nargin<4
                confidence = obj.DefaultConfidenceLevel;
            end
            if nargin<5
                [Observed_lb, observed_ub] = obj.call('errorBoundsObserved',image, theta_mle, confidence);
            else
                [Observed_lb, observed_ub] = obj.call('errorBoundsObserved',image, theta_mle, confidence, obsI);
            end
        end

        function [expected_lb, expected_ub] = errorBoundsExpected(obj, theta_mle, confidence)
            % [expected_lb, expected_ub] = obj.errorBoundsExpected(theta_mle, confidence)
            %
            % Compute the error bounds using the expected (Fisher) information at the MLE estimate.  This
            % is independent of the image, assuming Gaussian error with variance given by CRLB.
            %
            % (in) theta_mle: the theta MLE's to estimate error bounds for. size:[NumParams,N]
            % (in) confidence: (optional)  desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (out) expected_lb: the expected-information-based confidence interval lower bound for parameters for each theta, size:[NumParams,N]
            % (out) expected_ub: the expected-information-based confidence interval upper bound for parameters for each theta, size:[NumParams,N]
            theta_mle = obj.checkTheta(theta_mle);
            if nargin<3
                confidence = obj.DefaultConfidenceLevel;
            end
            [expected_lb, expected_ub] = obj.call('errorBoundsExpected', theta_mle, confidence);
        end

        function [profile_lb, profile_ub, theta_mle_out] = errorBoundsProfileLikelihood(obj, image, theta_mle, confidence, estimator_algorithm, params_to_estimate)
            % [profile_lb, profile_ub, theta_mle_out] = obj.errorBoundsProfileLikelihood(image, theta_mle, confidence, estimator_algorithm, params_to_estimate)
            %
            % Compute the error bounds using the profile-likelihood-based interval,
            %
            % (in) image: a stack of N images
            % (in) theta_mle: (optional) a stack of MLE/MAP estimated thetas size:[NumParams,N].  If not provided or set to invalid values the, it will be estimated first.
            % (in) confidence: (optional)  desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
            % (in) estimator_algorithm: (optional) name from obj.EstimationMethods.  The optimization method. [default=DefaultEstimatorMethod]
            % (in) params_to_estimate: (optional) boolean vector size:[NumParams,1], true if this parameter should have it's error estimated.
            %                                     This allows disable estimation of parameters that are not of interest.
            % (out) profile_lb: the profile-likelihood-based confidence interval lower bound for parameters, size:[NumParams,N]
            % (out) profile_ub: the profile-likelihood-based confidence interval upper bound for parameters, size:[NumParams,N]
            % (out) theta_mle: (optional) the theta_mle estimate, if not already produced. size:[NumParams,N]

            image = obj.checkImage(image);
            theta_mle = obj.checkTheta(theta_mle);
            if nargin<4
                confidence = obj.DefaultConfidenceLevel;
            end
            if nargin<5
                estimator_algorithm = obj.DefaultEstimatorMethod;
            end
            if nargin<6
                params_to_estimate = ones(obj.NumParams,1);
            end
            [profile_lb, profile_ub, theta_mle_out] = obj.call('errorBoundsProfileLikelihood',image, theta_mle, confidence, estimator_algorithm, params_to_estimate);
        end

        function varargout = errorBoundsPosteriorCredible(obj, image, theta_mle, confidence, num_samples, burnin, thin)
            % [credible_lb, credible_ub, posterior_sample] = obj.errorBoundsPosteriorCredible(image, theta_mle, confidence, max_samples)
            %
            % Compute the error bounds using the posterior credible interval, estimated with an mcmc sampling, using at most max_samples
            %
            % (in) image: a stack of n images to estimate
            % (in) theta_mle: (optional) Initial theta guesses as the MLE estimate size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (in) confidence: (optional) desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
            % (in) num_samples: (optional) Number of (post-filtering) posterior samples to aquire. [default=obj.DefaultMCMCNumSamples]
            % (in) burnin: (optional) Number of samples to throw away (burn-in) on initialization [default=obj.DefaultMCMCBurnin]
            % (in) thin: (optional) Keep every # samples. Value of 0 indicates use the model default. This is suggested.
            %                       When thin=1 there is no thinning.  This is also a good option if rejections are rare. [default=obj.DefaultMCMCThin]
            % (out) posterior_mean: (optional) size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            % (out) credible_lb: (optional) size:[NumParams,n] posterior credible interval lower bounds for each parameter for each image
            % (out) credible_ub: (optional) size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            [varargout{1:nargout}] = obj.estimatePosterior(image, theta_mle, confidence, num_samples, burnin, thin);
        end

        function [theta, varargout] = estimate(obj, image, estimator_algorithm, theta_init)
            % [theta, obsI, llh, stats] = obj.estimate(image, estimator_algorithm, theta_init)
            %
            % Use a multivariate constrained optimization algorithm to estimate the model parameter theta for each of
            % a stack of n images.  Returns observed information and log-likelihood as optional parameters.
            %
            % This is the maximum-likelihood estimate for 'MLE' models and the maximum a posteriori estimate for 'MAP' models.
            %
            % 'Newton' or 'TrustRegion' are suggested as the most robust estimators.
            %
            % (in) image: a stack of N images
            % (in) estimator_algorithm: (optional) name from obj.EstimationMethods.  The optimization method. [default=DefaultEstimatorMethod]
            % (in) theta_init: (optional) Initial theta guesses size:[NumParams, N].  Values of 0 indicate
            %            that we have no initial guess and the estimator should form its own guess.
            % (out) theta: size:[NumParams, N] estimated theta maximum for each image.
            % (out) rllh: (optional) size:[1,N] double of the relative log likelihood at each theta estimate.
            % (out) obsI: (optional) size:[NumParams,NumParams, N] the observed information at the MLE for each image.
            % (out) stats: (optional) A 1x1 struct of fitting statistics.
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
            switch estimator_algorithm
                case 'GPUGauss'
                    [theta, varargout{1:nargout-1}] = obj.estimate_GPUGaussMLE(image);
                case 'matlab-fminsearch'
                    [theta, varargout{1:nargout-1}] = obj.estimate_fminsearch(image, theta_init);
                case {'matlab-quasi-newton','matlab-trust-region-reflective','matlab-trust-region','matlab-interior-point'}
                    [theta, varargout{1:nargout-1}] = obj.estimate_toolbox(image, theta_init, estimator_algorithm);
                otherwise
                    [theta, varargout{1:nargout-1}] = obj.call('estimate',image, estimator_algorithm, theta_init);
            end
        end

        function varargout = estimateProfileLikelihood(obj, image, fixed_parameters, fixed_values, estimator_algorithm, theta_init)
            % [profile_likelihood, profile_parameters, stats]  = obj.estimateProfileLikelihood(image, fixed_parameters, fixed_values, estimator_algorithm, theta_init)
            %
            % Compute the profile likelihood for a single image and single parameter, over a range of
            % values.  For each value, the parameter of interest is fixed and the other parameters are
            % optimized with the estimator_algorithm in parallel with OpenMP.
            %
            % (in) image: a single images
            % (in) fixed_parameters: uint64 [NParams,1] 0=free 1=fixed.  At least one paramer must be fixed and at least one parameter must be free.
            % (in) fixed_values: size:[NumFixedParams,N], a vector of N values for each of the fixed parameters at which to maximimize the other (free) parameters at.
            % (in) estimator_algorithm: (optional) name for the optimization method. (default = 'TrustRegion') [see: obj.EstimationMethods]
            % (in) theta_init: (optional) Initial theta guesses size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                  If only a single parameter [NumParams,1] is given, each profile estimation will use this single theta_init.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            % (out) profile_likelihood: size:[N,1] profile likelihood for the parameter at each value.,
            % (out) profile_parameters: (optional) size:[NumParams,N] parameters that achieve the profile likelihood maximum at each value.
            % (out) stats: (optional) Estimator stats dictionary.
            image = obj.checkImage(image);
            if nargin<5
                estimator_algorithm = obj.DefaultEstimatorMethod;
            end
            if nargin<6
                theta_init = [];
            end
            fixed_parameters = uint64(fixed_parameters~=0);
            Nfixed = sum(fixed_parameters);
            if Nfixed<1 || Nfixed>obj.NumParams
                error('MappelBase:InvalidValue','fixed_parameters should have at least one fixed and at least one free parameter');
            end
            fixed_values_sz = size(fixed_values);
            if fixed_values_sz(1) ~= Nfixed
                error('MappelBase:InvalidSize','fixed_values must have one row for each parameter indicated in fixed_parameters');
            end
            theta_init = obj.checkThetaInit(theta_init, fixed_values_sz(2)); %expand theta_init to size:[NumParams,n]

            [varargout{1:nargout}] = obj.call('estimateProfileLikelihood',image, fixed_parameters, fixed_values, estimator_algorithm, theta_init);
        end

        function varargout = estimateDebug(obj, image, estimator_algorithm, theta_init)
            % [theta, obsI, llh, stats, sample, sample_rllh] = estimatedebug(image, estimator_algorithm, theta_init)
            %
            % Debugging routine.  Works for a single image.  Returns entire sequence of evaluated points and their llh.
            % The first entry of the evaluated_seq is theta_init.  The last entry may or may not be
            % theta_est.  It is strictly a sequence of evaluated thetas so that the lenght of the
            % evaluated_seq is the same as the number of RLLH evaluations performed by the maximization
            % algorithm.
            %
            % (in) image: a size:[flip(ImageSize)] image
            % (in) estimator_algorithm: (optional) name from obj.EstimationMethods.  The optimization method. [default=DefaultEstimatorMethod]
            % (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
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
                case 'matlab-fminsearch'
                    [varargout{1:nargout}] = obj.estimateDebug_fminsearch(image, theta_init);
                case {'matlab-quasi-newton','matlab-trust-region-reflective','matlab-trust-region','matlab-interior-point'}
                    [varargout{1:nargout}] = obj.estimateDebug_toolbox(image, theta_init, estimator_algorithm(8:end));
                otherwise
                    [varargout{1:nargout}] = obj.call('estimateDebug',image, estimator_algorithm, theta_init);
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
            % post-processing, along with the rllh at each sample.  Optional arguments are only computed
            % if required.
            %
            % MCMC sampling can be controlled with the optional num_samples, burnin, and thin arguments.
            %
            % The confidence parameter sets the confidence-level for the credible interval bounds.  The
            % credible intervals bounds are per-parameter, i.e, each parameter at index i is individually
            % estimated to have a credible interval from lb(i) to ub(i), using the sample to integrate
            % out the other parameters.
            %
            % (in) image: a stack of n images to estimate
            % (in) theta_init: (optional) Initial theta guesses size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (in) confidence: (optional) desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
            % (in) num_samples: (optional) Number of (post-filtering) posterior samples to aquire. [default=obj.DefaultMCMCNumSamples]
            % (in) burnin: (optional) Number of samples to throw away (burn-in) on initialization [default=obj.DefaultMCMCBurnin]
            % (in) thin: (optional) Keep every # samples. Value of 0 indicates use the model default. This is suggested.
            %                       When thin=1 there is no thinning.  This is also a good option if rejections are rare. [default=obj.DefaultMCMCThin]
            % (out) posterior_mean: size:[NumParams,n] posterior mean for each image
            % (out) credible_lb: (optional) size:[NumParams,n] posterior credible interval lower bounds for each parameter for each image
            % (out) credible_ub: (optional) size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
            % (out) posterior_cov: (optional) size:[NumParams,NumParams,n] posterior covariance matrices for each image
            % (out) mcmc_samples: (optional) size:[NumParams,max_samples,n] complete sequence of posterior samples generated by MCMC for each image
            % (out) mcmc_samples_rllh: (optional) size:[max_samples,n] relative log likelihood of sequence of posterior samples generated by MCMC. Each column corresponds to an image.

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
            % well as the candidate sequence. Does not do burnin or thinning.
            %
            % (in) image: a sinle images to estimate
            % (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
            %                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
            %                   should be auto estimated.
            % (in) num_samples: (optional) Number of (post-filtering) posterior samples to aquire. [default=obj.DefaultMCMCNumSamples]
            % (out) sample: A size:[NumParams,num_samples] array of thetas samples
            % (out) sample_rllh: A size:[1,num_samples] array of relative log likelihoods at each sample theta
            % (out) candidates: (optional) size:[NumParams, num_samples] array of candidate thetas
            % (out) candidate_rllh: (optional) A size:[1, num_samples] array of relative log likelihoods at each candidate theta
            image=obj.checkImage(image);
            if nargin<3
                theta_init = [];
            end
            theta_init = obj.checkThetaInit(theta_init, 1);
            if nargin<4 || isempty(num_samples) || num_samples<=1
                num_samples = obj.DefaultMCMCMaxSamples;
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

        %%Model Testing
        function [llh, theta_bg_mle] = uniformBackgroundModelLLH(obj, ims)
            % Test the model fit of a 1-parameter constant background model to the stack of images.
            % The mle estimate for a 1-parameter baground parameter is just the mean of the image.
            % The log-likelihood is calculated at this MLE estimate.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (out) llh: a length N vector of the LLH for each image for the consant-background model
            % (out) theta_bg_mle: a length N vector of the estimated MLE constant background.
            npixels = prod(obj.ImageSize);
            ims = reshape(ims,npixels,[]);
            theta_bg_mle = mean(ims)';
            llh =  log(theta_bg_mle).*sum(ims)' - npixels*theta_bg_mle - sum(gammaln(ims+1))';
        end

        function [pass, LLRstat] = modelComparisonUniform(obj, alpha, ims, theta_mle)
            % Do a LLH ratio test to compare the emitter model to a single parameter constant background model
            % The images are provided along with the estimated theta mle values for the emitter model.
            % The LLH ratio test for nested models can be used and we compute the test statistice
            % LLRstat = -2*llh_const_bg_model + 2*llh_emitter_model.  This should be chisq distributed
            % with number of degrees of freedom given by obj.NumParams-1 since the const bg model has 1
            % param.
            % (in) alpha: 0<=alpha<1 - the certainty with which we should be sure to accept an emitter fit
            %                          vs. the uniform background model.  Values close to 1 reject more
            %                          fits.  Those close to 0 accept most fits.  At 0 only models where
            %                          the constant bg is more likely (even though it has only 1 free parametet)
            %                          will be rejected.  These arguably should always be rejected.  It
            %                          indicates an almost certainly bad fit.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (in) theta_mle: a double size:[NumParams, n] sequence of theta MLE values for each image in
            %                 ims
            % (out) pass: a boolean length N vector which is true if the emitter model passes for this test.
            %             in otherwords it is true for images with good fits that should be kept.  Images
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
            % In this model each pixel has its own parameter and that pixels mle will of course be the
            % value of the pixel itself.  Unlike the constant bg model there is no point to return the
            % mle values themselves since they are just the images.
            % (in) ims: a double size:[flip(ImageSize), n] image stack
            % (out) llh: a length N vector of the LLH for each image for the consant-background model
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
                theta_est=obj.estimatePosterior(ims,count,theta_init);
            else
                theta_est=obj.estimate(ims,estimator, theta_init);
            end
            error = theta_est-repmat(theta,1,nTrials);
            rmse = sqrt(mean(error.*error,2));
            stddev = std(error,0,1);
        end

        function [theta_est, obsI]=evaluateEstimatorOn(obj, estimator, images)
            %Evaluate this 2D estimator at a particular theta using the given samples which may have
            % been generated using different models or parameters.
            %
            % (in) estimator - String. Estimator name.  Can be any of the MAP estimator names or 'Posterior N' where N is a count
            % (in) images - size[flip(ImageSize),N] -  An array of sample images to test on
            % (out) theta_est - size:[NumParams,N]: the estimated thetas
            % (out) est_var - size:[NumParams,N]: the estimated variance of the estimate at each theta
            if strncmpi(estimator,'posterior',9)
                count = str2double(estimator(10:end));
                [theta_est,est_cov]=obj.estimatePosterior(images,count);
                est_var=zeros(obj.NumParams,size(est_cov,3));
                for i=1:size(est_cov,3)
                    est_var(:,i)=diag(est_cov(:,:,i));
                end
            else
                [theta_est,est_var]=obj.estimate(images,estimator);
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
            % are irrelevent as they will be modified for each grid point.
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
            % error as a gaussian
            % (in) theta - The parameter value to visualize
            % (in) theta_err - (optional) The RMSE error or sqrt(CRLB) that represents
            %                   the localization error. [default = run a simulation to estimate error] 
            % (in) res_factor - [optinal] integer>1 - the factor to blow up the image
            %                   [default = 100]
            % (out) srim - double size:[sizeY*res_factor,sizeX*res_factor]
            %             A super-res rendering of the emitter fit postion.  
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

    methods (Access = public)
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

        function [theta,  llh, obsI, stats]=estimate_fminsearch(obj, image, theta_init)
            %
            % Uses matlab's fminsearch (Simplex Algorithm) to maximize the LLH for a stack of images.
            % This is available in the core Matlab and does not require the optimization toolbox
            %
            % Uses LLH function evaluation calculations from C++ interface.
            % (in) image - a stack of N double images size:[flip(ImageSize), n]
            % (in) theta_init: (optional) Initial theta guesses size (NumParams x n).  Values of 0 indicate
            %            that we have no initial guess and the estimator should form its own guess.
            % (out) theta: size:[NumParams, N] double of estimated theta values
            % (out) llh: (optional) a size:[1, N] double of the log likelihood at each theta estimate.
            % (out) obsI: (optional) size:[NumParams, NumParams, N] estimate of the CRLB for each parameter estimate.
            %             This gives the approximate variance in the theta parameters
            % (out) stats: (optional) A 1x1 struct of fitting statistics.
            max_iter=5000;
            N = size(image,3);
            if nargin<3 || isempty(theta_init)
                theta_init = obj.estimate(image,'Heuristic');
            elseif isvector(theta_init)
                theta_init = repmat(theta_init',1,N);
            end
            opts = optimset('fminsearch');
            opts.MaxIter = max_iter;
            opts.Diagnostics = 'on';
            opts.Display = 'off';
            problem.solver = 'fminsearch';
            problem.options = opts;
            theta = zeros(obj.NumParams, N);
            llh = zeros(1,N);
            iterations = zeros(1,N);
            fevals = zeros(1,N);
            
            
            function val = callback(x)
                if any(x-obj.ParamLBound<1e-6) || any(obj.ParamUBound-x<1e-6)
                    %Method 1 - Inf
                    %val = inf;
                    %Method 2 - bound
                    eps = 1e-6;
                    x = min(max(obj.ParamLBound+eps,x),obj.ParamUBound-eps);
                    val = -obj.modelLLH(im,x);
                else
                    val = -obj.modelLLH(im,x);
                end
            end
            
            for n=1:N
                im = image(:,:,n);
                problem.objective = @callback;
                problem.x0 = theta_init(:,n);
                [theta(:,n), llh_opt, flag, out] = fminsearch(problem);
                llh(n) = -llh_opt;
                fevals(n) = out.funcCount;
                iterations(n) = out.iterations;
            end
            obsI = obj.observedInformation(image,theta);
            stats.method = out.algorithm;
            stats.iterations = out.iterations;
            stats.flag = flag;
            stats.iterations = iterations;
            stats.fevals = fevals;
        end

        function [theta, obsI, llh, sequence, sequence_llh,stats] = estimateDebug_fminsearch(obj, image, theta_init)
            max_iter=5000;
            sequence=zeros(obj.NumParams, max_iter);
            function stop = output(theta, opt, state)
                sequence(:,opt.iteration+1)=theta;
                stop = strcmp(state,'done');
            end
            if isempty(theta_init)
                theta_init = obj.estimate(image,'Heuristic', theta_init);
            end
            opts = optimset('fminsearch');
            opts.MaxIter = max_iter;
            opts.Diagnostics = 'on';
            opts.Display='final-detailed';
            opts.OutputFcn = @output;
            problem.solver = 'fminsearch';
            problem.options = opts;
            
            function val = callback(x)
                if any(x-obj.ParamLBound<1e-6) || any(obj.ParamUBound-x<1e-6)
                    val = inf;
                else
                    val = -obj.modelLLH(image,x);
                end
            end
            
            problem.objective = @callback;
            problem.x0 = theta_init(:);
            [theta, fval, flag, out] = fminsearch(problem);
            obsI = obj.observedInformation(image, theta);
            llh = -fval;
            stats.method = out.algorithm;
            stats.iterations = out.iterations;
            stats.flag = flag;
            sequence = sequence(:,1:out.iterations);
            in_bounds = obj.thetaInBounds(sequence);
            sequence_llh(in_bounds) = obj.modelRLLH(image, sequence(:,in_bounds));
            sequence_llh(~in_bounds) = -inf;
            stats.sequenceLen = size(sequence,2);
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
            % gradieant values.
            %
            % (in) image - A stack of N images. type: double size:[flip(ImageSize), N]
            % (in) theta_init - (optional) size:[NumParams, N] array giving initial theta value for each image
            %                   Default: Use Heuristic.
            % (in) algorithm - (optional) string: The algorithm to choose.
            %                    ['quasi-newton', 'interior-point', 'trust-region', 'trust-region-reflective']
            %                    [default: 'trust-region']
            % (out) theta - size:[NumParams, N]. Optimal theta value for each image
            % (out) llh - size:[NumParams, N]. LLH value at optimal theta for each image
            % (out) obsI - size:[NumParams, N]. CRLB value at each theta value.
            % (out) stats - statistics of fitting algorithm's performance.
            if nargin==3
                algorithm = 'trust-region';
            end
            if nargin==2 || isempty(theta_init)
                theta_init = obj.estimate(image,'Heuristic', theta_init);
            end
            switch algorithm
                case 'quasi-newton'
                    solver = @fminunc;
                case 'trust-region'
                    solver = @fminunc;
                case 'interior-point'
                    solver = @fmincon;
                case 'trust-region-reflective'
                    solver = @fmincon;
                otherwise
                    error('MappelBase:estimateDebug_fmincon','Unknown maximization method: "%s"', algorithm);
            end
            problem.solver = func2str(solver);
            opts = optimoptions(problem.solver);
            opts.Algorithm = algorithm;
            opts.SpecifyObjectiveGradient = true;
            opts.Diagnostics = 'on';
            opts.Display = 'iter-detailed';
            opts.MaxIterations = 500;
            switch algorithm
                case 'quasi-newton'
                    %Cannot supply hessian for quasi-newton
                    opts.HessUpdate = 'bfgs'; %Method to choose search direction. ['bfgs', 'steepdesc', 'dfp']
                    objective = @(im,theta) deal(-obj.LLH(im,theta), -obj.modelGrad(im,theta)); % 2 arg (obj,grad)
                case {'trust-region','trust-region-reflective'}
                    opts.HessianFcn = 'objective'; %Third arg in objective function.
                    opts.SubproblemAlgorithm = 'factorization'; % vs. 'cg' 
                    objective = @(im,theta) deal(-obj.LLH(im,theta), -obj.modelGrad(im,theta), -obj.modelHessian(im,theta));% 3 arg (obj,grad,hess)
                case 'interior-point'
                    objective = @(im,theta) deal(-obj.LLH(im,theta), -obj.modelGrad(im,theta)); % 2 arg (obj,grad)
            end
            problem.options = opts;
            nIms = size(image, obj.ImageDim+1); %number of images to process
            theta = zeros(obj.NumParams, nIms);
            llh = zeros(nIms,1);
            niters = zeros(nIms,1);
            flags = zeros(nIms,1);
            obsI = zeros(obj.NumParams,obj.NumParms,nIms);
            for n=1:nIms
                im = image(:,:,n);
                problem.x0 = theta_init(:,n);
                problem.objective = @(theta) objective(im,theta);
                switch algorithm
                    case 'interior-point'
                        problem.options.HessianFcn = @(theta, ~) -obj.modelHessian(im,theta);
                end
                [theta(:,n), llh(n), flags(n), out] = solver(problem);
                niters(n) = out.iterations;
                obsI(:,:,n) = obj.observedInformation(im,theta(:,n));
            end
            llh = -llh; %Flip llh objective values back to standard units
            stats.algorithm = algorithm;
            stats.iterations = niters;
            stats.flags = flags;
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
            % gradieant values.
            %
            % (in) image - A single image. type: double size:[flip(ImageSize)]
            % (in) theta_init - (optional) size:[NumParams, 1] initial theta value for image
            %                   Default: Use Heuristic.
            % (in) algorithm - (optional) string: The algorithm to choose.
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
            if nargin==3
                algorithm = 'trust-region';
            end
            if nargin==2 || isempty(theta_init)
                theta_init = obj.estimate(image,'Heuristic', theta_init);
            end
            switch algorithm
                case 'quasi-newton'
                    solver = @fminunc;
                case 'trust-region'
                    solver = @fminunc;
                case 'interior-point'
                    solver = @fmincon;
                case 'trust-region-reflective'
                    solver = @fmincon;
                otherwise
                    error('MappelBase:estimateDebug_fmincon','Unknown maximization method: "%s"', algorithm);
            end
            problem.solver = func2str(solver);
            opts = optimoptions(problem.solver);
            opts.Algorithm = algorithm;
            opts.SpecifyObjectiveGradient = true;
            opts.Diagnostics='on';
            opts.Display='final-detailed';
            
            max_iter=5000;
            sequence=zeros(obj.NumParams, max_iter);
            opts.MaxIterations = max_iter;
            function stop = output_func(theta, opt, state)
                sequence(:,opt.iteration+1)=theta;
                stop = strcmp(state,'done');
            end
            opts.OutputFcn = @output_func;
            
            switch algorithm
                case 'quasi-newton'
                    %Cannot supply hessian for quasi-newton
                    opts.HessUpdate = 'bfgs'; %Method to choose search direction. ['bfgs', 'steepdesc', 'dfp']
%                     problem.objective = @(theta) deal(-obj.LLH(image,theta), -obj.modelGrad(image,theta)); % 2 arg (obj,grad)
                case {'trust-region','trust-region-reflective'}
                    opts.HessianFcn = 'objective'; %Third arg in objective function.
                    opts.SubproblemAlgorithm = 'factorization'; % vs. 'cg' 
%                     problem.objective = @(theta) deal(-obj.LLH(image,theta), -obj.modelGrad(image,theta), -obj.modelHessian(image,theta));% 3 arg (obj,grad,hess)
                case 'interior-point'
                    opts.HessianFcn = @(theta, ~) -obj.modelHessian(image,obj.boundTheta(theta));
%                     problem.objective = @(theta) deal(-obj.LLH(image,theta), -obj.modelGrad(image,theta)); % 2 arg (obj,grad)
            end
            function varargout = objective(x)
                if obj.thetaInBounds(x)
                    [varargout{1:nargout}] = obj.modelObjective(image,x,true);
                else
                    varargout{1}=inf;
                    if nargout>1
                        %TODO: figure out how to make derivatives smooth
                        [~,grad,hess] = obj.modelObjective(image,obj.boundTheta(x),true); % Negation true
                        fprintf('Theta out of bounds; %s', x)
                        varargout{2}=grad;
                        if nargout==3
                            varargout{3}=hess;
                        end
                    end
                end
            end
            a=objective(theta_init);
            problem.objective = @objective;
            problem.options = opts;
            problem.x0 = theta_init;
            [theta, llh, stats.flag, stats.out] = solver(problem);
            obsI = obj.observedInformation(image, theta);
            llh = -llh; %Flip llh objective values back to standard units
            stats.algorithm = algorithm;
            stats.iterations = stats.out.iterations;
            sequence(:,stats.iterations+1:end)=[];
            sequence_llh = obj.modelRLLH(image,sequence);
        end


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
