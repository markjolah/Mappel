% A short test case, that shows how to use each operation.
% This is a functionality test not a unit test.
function quick_test(model_names,estimator_names,Nsample)
    if nargin<1
        model_names = {'Gauss2DMAP'};
    end
    if nargin<2
        estimator_names = {'Heuristic','CGaussHeuristic'};
    end
    if nargin<3
        Nsample = 1000;
    end
    for name_cell = model_names
        class_name = ['Mappel.' name_cell{1}];
        dim = eval([class_name '.ImageDim']);
        if dim==2
            im_size = [7,9];  %[sizeX, sizeY] size of an image: X is horizontal or #cols, Y is vertical or #rows
            psf_sigma = [0.9, 1.1]; %[psf_sigmaX,psf_sigmaY] Gaussian point-spread-function model.  Sigma in pixels.
            model_constructor = str2func(class_name);
            M = model_constructor(im_size,psf_sigma);
            disp(M.getStats())
            disp('HyperparamNames: '); disp(M.HyperparamNames');
            disp('Hyperparams: '); disp(M.Hyperparams');
            
            theta = M.samplePrior();
            thetas = M.samplePrior(Nsample);
            im = M.simulateImage(theta); % A single image sampled at theta
            im_stack = M.simulateImage(theta,Nsample); % A stack of images all sampled at theta
            im_thetas = M.simulateImage(thetas); % A stack of images sampled at each of Nsample different thetas.
            
            llh = M.modelLLH(im,theta);
            llh_stack = M.modelLLH(im_stack,theta);
            llh_thetas = M.modelLLH(im_thetas,thetas);

            grad = M.modelGrad(im,theta);
            grad_stack = M.modelGrad(im_stack,theta);
            grad_thetas = M.modelGrad(im_thetas,thetas);

            hess = M.modelHessian(im,theta);
            hess_stack = M.modelHessian(im_stack,theta);
            hess_thetas = M.modelHessian(im_thetas,thetas);

            for estimator_cell = estimator_names
                estimator = estimator_cell{1};
                est = M.estimateMax(im);
                est = M.estimateMax(im,estimator);
                est = M.estimateMax(im_stack,estimator);

                [est, llh] = M.estimateMax(im,estimator);
                [est, llh, obsI] = M.estimateMax(im,estimator);
                [est, llh, obsI, stats] = M.estimateMax(im,estimator);
                [est, llh, obsI, stats] = M.estimateMax(im_thetas,estimator);

                [est, llh, obsI, stats] = M.estimateMax(im, estimator, theta);
                [est, llh, obsI, stats] = M.estimateMax(im_stack, estimator,theta);
                [est, llh, obsI, stats] = M.estimateMax(im_thetas, estimator,thetas);

            end
        end
    end
end


