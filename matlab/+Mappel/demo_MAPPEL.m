
% MAPPEL Demo Script
% Mark J. Olah (mjo@cs.unm.edu)
% 03-2014
%
% MAPPEL == Maximum a-posteriori Point Emitter Localizaion
%
% MAPPEL is an object-oriented interface to C++ localization code for
% gaussian point-emitter localization with a poisson noise model.
%
% At the top-level MAPPEL consists of several classes each with a common
% interface, but that does a different type of fitting.
%
% The most common fit modes are represented by these classes:
%  Gauss2DMAP - 2D Maximum A-Posteriori fitting for a model with [X Y I bg]
%  Gauss2DsMAP - 2D Maximum A-Posteriori fitting for a model with [X Y I bg sigma]

% {{{ IMPORTANT: A note on image sizes and the order of x & y. }}}
% All parameters in MAPPEL are always listed in the order of [X, Y].  This
% is true for size and psf_sigma and the positions of the x and y variables
% in a theta parameter value.  However, the common interpretation of images
% in both Matlab and DipImage is to have an image be respresented with X as
% the column index and Y as the row index.  Since the matlab "size"
% function always returns [#rows, #cols], an image will appear to have size
% [sizeY, sizeX].  While this seems confusing, it makes an image more
% closely match the way we draw plots with the x-axis horizontal and the
% y-axis veritical, and this is how dip-image and matlab's 'imagesc'
% commands work.  Note that this is different than "GPUGaussMLE" code,
% which represents images in the opposite way.
%
% See: MAPPEL-UserGuid.pptx for more info

%imsize - This is the image size of the subregions.  Format is [sizeX sizeY].
imsize=[8,8];
psf_sigma=[1,1]; %symmetric PSF 

g2d = Gauss2DsMAP(imsize, psf_sigma);  % Use a model that also fits for the apparent psf sigma size for Out-of-focus emitters

%% Sampling thetas and making model images (optional)
N = 1e4; 
thetas = g2d.samplePrior(N); %Sample N points from the prior distribution
model_ims = g2d.modelImage(thetas); %Make a 3d stack of model images (these are the expected value for each pixel)

%% Run a fitting test
N = 1e5; 
test_theta = [2.718 4.321 1000 3 1.2]'; % [x y I bg sigma]
actual_CRLB = g2d.CRLB(test_theta); % Compute the actual CRLB at test_theta
sim_ims = g2d.simulateImage(test_theta, N); %Sample N realizations of the test_theta

tic;
[etheta, crlb, llh, stats] = g2d.estimateMAP(sim_ims); %Use Newton's method to find MAP estimates for each of a stack of images
%crlb - Cramer-Rao lower bound (this is in units matching variance, take the sqrt for the standard error)
%llh - The log-liklihood with all constant correction terms included (this will not match cGaussMLE which drops constant factors)
%stats - [optional] a struct with some statistics on the fitting.
fit_time=toc;
rmse = sqrt(mean((repmat(test_theta,1,N)-etheta).^2,2)); % compute root-mean-squared error for the estimator at test_theta with N samples

fprintf('Fit %i points in %f seconds. (%g fits/s)\n',N,fit_time, N/fit_time);
disp('TestTheta:')
disp(num2str(test_theta'));
disp('sqrt(CRLB(TestTheta)):');
disp(num2str(sqrt(actual_CRLB)'));
disp('RMSE(TestTheta):')
disp(num2str(rmse'));

%% Make a super resolution model image showing the uncertainty at this theta (optional)
%rmse = g2d.evaluateEstimatorAt('Newton',test_theta); %This does exactly the same error estimation as done above, but in a more convenient package
%sr_im = g2d.superResolutionModel(test_theta, rmse); %This is just for convenience and is not a fast SR code
%g2d.viewDipImage(sr_im);

%% Posterior estimation.  This is much slower, and not likely needed for 2D fitting.
%[etheta, ecov] = g2d.estimatePosterior(sim_ims);
% etheta is the mean of the sampled points
% cov is the sample covariance matrix.  Take sqrt(diag(cov)) to get a
% standard deviation for the parameters.

