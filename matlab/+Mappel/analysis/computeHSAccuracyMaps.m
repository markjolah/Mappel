function Data=computeHSAccuracyMaps()

    Data.sz=[8 10 14];
%     Data.lambda=linspace(500, 550, Data.sz(3)+1);
%     Data.psf=[1.0 1.2 1.5*(50./14)] ;
    Data.lambda=0:Data.sz(3);
    Data.psf=[1.3 1.2 2.5] ;
    Data.gridsize=[1 1]*50;
    Data.trials=100;
    Data.generativeModel=BlinkHSsMAP(Data.sz,Data.psf,Data.lambda);
    Data.model{1}=BlinkHSsMAP(Data.sz,Data.psf,Data.lambda);
    Data.model{2}=GaussHSsMAP(Data.sz,Data.psf,Data.lambda);
    Data.model{3}=GaussHSMAP(Data.sz,Data.psf,Data.lambda);
    Data.estimator{1}='Huristic';
    Data.estimator{2}='Newton';
    Data.estimator{3}='Posterior 10000';
    Data.theta=[4 5 7 3000 2 1.0 2.5];
    Data.blinkpattern=ones(1,Data.sz(1));
%     Data.blinkpattern=[0 0 0.3 1 1 1 0.5 0.2];
%     Data.blinkpattern=[0.7 0.4 1 0.9 0 0.8 1 1];
    Data.blinkpattern=[1 1 1 0.9 0.1 1 1 1];
    Data.theta=[Data.theta Data.blinkpattern];
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
