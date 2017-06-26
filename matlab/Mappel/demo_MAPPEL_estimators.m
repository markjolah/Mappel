
% MAPPEL Estimators Demo Script
% Mark J. Olah (mjo@cs.unm.edu)
% 03-2014
%
% Demo new fitting methods, features and speeds
%

%imsize - This is the image size of the subregions.  Format is [sizeX sizeY].
imsize=[9,9];
psf_sigma=[1,1]; %symmetric PSF 

g2d = Gauss2DMAP(imsize, psf_sigma);  % Use a model that also fits for the apparent psf sigma size for Out-of-focus emitters

%% Sampling thetas and making model images (optional)
N = 1e1; %number of sample images to run on 
theta = [3.8 4.2 100000 100]; % A high intensity emitter

fprintf('Theta:     %s\n',num2str(theta,'%12.6g'));
fprintf('Heruistic: %s\n',num2str(g2d.estimateMAP(g2d.simulateImage(theta) ,'Heuristic')','%12.6g') );

tic;
estimator='Newton';
theta_init=[];
err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
T=toc;
fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

tic;
estimator='Newton';
theta_init=theta;
err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
T=toc;
fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

tic;
estimator='NewtonRaphson';
theta_init=[];
err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
T=toc;
fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

tic;
estimator='NewtonRaphson';
theta_init=theta;
err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
T=toc;
fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

if (N<=1000)
    %These methods are the code from Smith et. al. Nature Methods (2010)
    %Speed is limited and the number of iterations should be tuned.
    %The "CGauss" method runs ins parallel on the CPU but has a hard-coded
    %number of iterations ~20 or so.
    %The "GPUGauss" method requires CUDA 7.0 and has a tunable iteration
    %count.
    tic;
    estimator='CGauss';
    theta_init=[];
    err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
    T=toc;
    fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
    fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
    fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
    fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

    tic;
    estimator='GPUGauss';
    g2d.GPUGaussMLE_Iterations=500; %Need to tune this for your theta value
    theta_init=[];
    err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
    T=toc;
    fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
    fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
    fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
    fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));
end

if (N<=10)
    %These methods use the matlab optimization toolbox and are much slower.
    %They should be run with only a few images or they will take forever to
    %finish.
    
    % interior-point method from fmincon
    tic;
    estimator='interior-point';
    theta_init=[];
    err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
    T=toc;
    fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
    fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
    fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
    fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));

    % trust-region method from fmincon
    tic;
    estimator='trust-region';
    theta_init=[];
    err=g2d.evaluateEstimatorAt(estimator, theta, N, theta_init);
    T=toc;
    fprintf('Estimator: %s  #Samples: %i  ThetaInit: %s\n',estimator,N,mat2str(theta_init));
    fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,N/T);
    fprintf('-->sqrt(MSE):  %s\n',num2str(err','%12.6g'));
    fprintf('-->sqrt(CRLB): %s\n\n',num2str(sqrt(g2d.CRLB(theta))','%12.6g'));
end