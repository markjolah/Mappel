
function Data=pointCompareEstimators()
    Data=struct();
    Data.Count=50;
    Data.Size=8;
    Data.PSFSigma=1.0;
    % Data.Theta=[5.0, 3.5, 2000, 10, 1.0, 1, 1,1,0.1,0,0.5, 1, 1]';
    Data.Theta=[3.0, 5.8, 1000, 5, 1.9]';
    Data.blinkPattern=ones(8,1);
%     Data.blinkPattern=[1 1 0.2 0 0.8 0 0.9 1]';
    genModel=Blink2DsMAP(Data.Size,Data.PSFSigma);
    model=Gauss2DsMAP(Data.Size,Data.PSFSigma);
%     model=Gauss2DMAP(Data.Size,Data.PSFSigma);
    Data.genTheta=[Data.Theta; Data.blinkPattern];
    Data.model=model;
    Data.genModel=genModel;
    Data.ModelName=model.Name;
    % Data.Methods={'Huristic', 'Newton', 'NewtonRaphson', 'QuasiNewton', 'Posterior'};
    % Data.Methods={'Huristic','CGauss' 'Newton', 'NewtonRaphson', 'QuasiNewton', 'Posterior'};
    Data.Methods={'SimulatedAnnealing','NewtonRaphson', 'Newton', 'Posterior 2000'};
    Data.nMethods=length(Data.Methods);
    Data.nIs=40;
    Data.minI=50;
    Data.maxI=3000;
    Data.Is=linspace(Data.minI,Data.maxI,Data.nIs);
    Data.rmse=cell(1,Data.nMethods);
    Data.bias=cell(1,Data.nMethods);
    Data.var=cell(1,Data.nMethods);
    Data.mean_llh=cell(1,Data.nMethods);
    Data.CRLB=zeros(Data.model.nParams,Data.nIs);
    Data.mean_est_var=cell(1,Data.nMethods);
    test_thetas=repmat(Data.Theta,1,Data.nIs);
    test_thetas(3,:)=Data.Is;

    for m=1:Data.nMethods
        name=Data.Methods{m};
        Data.bias{m}=zeros(model.nParams, Data.nIs);
        Data.var{m}=zeros(model.nParams, Data.nIs);
        Data.mean_est_var{m}=zeros(model.nParams, Data.nIs);
        Data.rmse{m}=zeros(model.nParams, Data.nIs);
        Data.mean_llh{m}=zeros(1, Data.nIs);
        for i=1:Data.nIs
            theta=Data.Theta;
            theta(3,:)=Data.Is(i);
            genTheta=Data.genTheta;
            genTheta(3,:)=Data.Is(i);
            thetas=repmat(theta,1,Data.Count);
            ims=genModel.simulateImage(genTheta,Data.Count);
            Data.CRLB(:,i)=model.CRLB(theta);
            fprintf('Method(%i) Name: %s Intensity: %f\n',m,name, Data.Is(i));
            parts=strsplit(name);
            if (length(parts)==2)
                [theta_est,post_cov]=model.estimatePosterior(ims,str2double(parts{2}));
                llh_est=model.LLH(ims,theta_est);
                mean_post_var=zeros(model.nParams,1);
                for k=1:Data.Count
                    mean_post_var=mean_post_var+diag(post_cov(:,:,k));
                end
                mean_post_var=mean_post_var/Data.Count;
                Data.mean_est_var{m}(:,i)=mean_post_var;
            else
                [theta_est, crlb_est, llh_est, stats]=model.estimateMAP(ims,name);
                Data.mean_est_var{m}(:,i)=mean(crlb_est,2);
            end
            error=thetas-theta_est;
            Data.bias{m}(:,i)=mean(error,2);
            Data.var{m}(:,i)=var(theta_est,[],2);
            Data.rmse{m}(:,i)=sqrt(mean(error.*error,2));
            Data.mean_llh{m}(i)=mean(llh_est);
        end
    end
end
