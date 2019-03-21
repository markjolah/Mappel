
% Mappel Estimators Demo Script
% Mark J. Olah (mjo@cs.unm.edu)
% 2014-2019
%

function demoEstimators(model, Nsamples,theta)
    if nargin<1
        model = Mappel.Gauss2DMAP([9,9],[1.1,1.1]);
    end
    if nargin<2
        Nsamples = 1e4;
    end
    if nargin<3
        theta = model.samplePrior();
    end

    ims = model.simulateImage(theta,Nsamples);

    fprintf('Model: %s\n',model.Name);
    fprintf('ImageSize: %s\n',mat2str(model.ImageSize));
    fprintf('PSFSigmaMin: %s\n',mat2str(model.PSFSigmaMin));
    fprintf('PSFSigmaMax: %s\n',mat2str(model.PSFSigmaMax));
    fprintf('Theta:     %s\n',mat2str(theta',6));
    fprintf('sqrt(CRLB):     %s\n',mat2str(sqrt(model.CRLB(theta)),6));

    estimators = {'Heuristic','Newton','NewtonDiagonal','QuasiNewton','TrustRegion'};
    if(model.ImageSize(1)==model.ImageSize(2) && model.PSFSigmaMin(1)==model.PSFSigmaMin(2))
        estimators{end+1} = 'CGauss';
        if ispc()
            estimators{end+1} = 'GPUGauss';
        end
    end
    if Nsamples<=100
        estimators = [estimators {'matlab-fminsearch', 'matlab-quasi-newton','matlab-trust-region','matlab-interior-point'}];
    end

    for est_cell = estimators
        estimator = est_cell{1};
        tic;
        [est_thetas,rmse,covariance] = model.evaluateEstimatorOn(estimator, ims, theta);
        
        T=toc;
        fprintf('\nEstimator: %s  #Samples: %i\n',estimator,Nsamples);
        fprintf('-->Runtime: %.5fs  #Fits/Sec: %g\n',T,Nsamples/T);
        fprintf('-->sqrt(MSE):  %s\n',num2str(rmse','%12.6g'));
    end
end
