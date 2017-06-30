function Data=computeAccuracyMaps()

    Data.sz=[8 8];
    Data.psf=1.0;
    Data.gridsize=[1 1]*30;
    Data.trials=100;
    Data.generativeModel=Gauss2DsMAP(Data.sz,Data.psf);
%     Data.model{1}=Blink2DsMAP(Data.sz,Data.psf);
    Data.model{1}=Gauss2DsMAP(Data.sz,Data.psf);
    Data.model{2}=Gauss2DMAP(Data.sz,Data.psf);
    Data.model{3}=Gauss2DsMLE(Data.sz,Data.psf);
    Data.model{4}=Gauss2DMLE(Data.sz,Data.psf);
    Data.estimator{1}='Heuristic';
    Data.estimator{2}='Newton';
    Data.estimator{3}='NewtonRaphson';
%     Data.estimator{3}='Posterior 2000';
%     Data.estimator{4}='CGauss';
    Data.theta=[3.5 5 1000 5 1.0];
%     Data.blinkpattern=ones(1,Data.sz(1));
%     Data.blinkpattern=[0 0 0.3 1 1 1 0.5 0.2];
%    Data.blinkpattern=[0.7 0.4 1 0.1 0 0.8 1 1];
%     Data.blinkpattern=[1 1 1 0.9 0.3 0 0.4 1];
%     Data.theta=[Data.theta Data.blinkpattern];
    [Data.theta_grid, Data.sample_grid]=Data.generativeModel.makeThetaGridSamples(Data.theta,Data.gridsize,Data.trials);
    Data.theta_est_grid=cell(length(Data.model),length(Data.estimator));
    Data.est_var_grid=cell(length(Data.model),length(Data.estimator));
    for mi=1:length(Data.model)
        for ei=1:length(Data.estimator)
            [Data.theta_est_grid{mi,ei},Data.est_var_grid{mi,ei}]=...
                Data.model{mi}.mapEstimatorAccuracy(Data.estimator{ei}, Data.sample_grid);
        end
    end
end
