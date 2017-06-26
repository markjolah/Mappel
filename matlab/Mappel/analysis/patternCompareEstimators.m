
function Data=patternCompareEstimators()
    Data.Count=1000;
    Data.Size=8;
    Data.PSFSigma=1.0;
    Data.theta=[3.8, 5.4, 1000, 5, 1.8, ones(1,Data.Size)];

    Data.genModel=Blink2DsMAP(Data.Size,Data.PSFSigma);

    Data.Models={Blink2DsMAP(Data.Size,Data.PSFSigma), Gauss2DsMAP(Data.Size,Data.PSFSigma)};      
    Data.nModels=length(Data.Models);
    Data.Methods={'Heuristic', 'SimulatedAnnealing', 'NewtonRaphson', 'Posterior 50000'};
%     Data.Methods={'Newton'};
    Data.nMethods=length(Data.Methods);

    Data.nIs=35;
    Data.minI=100;
    Data.maxI=3000;
    Data.Is=round(linspace(Data.minI,Data.maxI,Data.nIs));
    Data.thetas=zeros(length(Data.theta),Data.Count,Data.nIs);
    Data.samples=zeros(Data.Size,Data.Size,Data.Count,Data.nIs);
    Data.rmse=cell(Data.nModels,Data.nMethods);
    Data.bias=cell(Data.nModels,Data.nMethods);
    Data.var=cell(Data.nModels,Data.nMethods);
    Data.mean_llh=cell(Data.nModels,Data.nMethods);
    Data.CRLB=cell(Data.nModels,1);
    Data.mean_est_err=cell(Data.nModels,Data.nMethods);
    Data.mean_est_var=cell(Data.nModels,Data.nMethods);
    Data.scan_rate=1e-3; %Line-scan rate in lines/s;
    Data.burnin=1e-2; %Length of burnin for simulation
    Data.qdsim=QDBlinkSim(0, Data.scan_rate,Data.Size,Data.burnin); %Use this to simulate blinking

    %Make the thetas and the samples
    for i=1:Data.nIs
        genTheta=Data.theta(:); %Use these to simulate images
        genTheta(3)=Data.Is(i);
        genThetas=repmat(genTheta,1,Data.Count);
        fprintf('Generating Samples I=%i (%i/%i)\n',Data.Is(i),i,Data.nIs);
        for j=1:Data.Count
            duty_ratio=Data.qdsim.simulateImage();
            while mean(duty_ratio)<0.5 %Must be at least 50% and less than full on
%             while mean(duty_ratio)<0.5 || mean(duty_ratio)>=0.98 %Must be at least 50% and less than full on
                duty_ratio=Data.qdsim.simulateImage();
            end
%             duty_ratio=[1 1 1 0.8 0 0.9 1 0.3];
%             duty_ratio=ones(1,Data.Size); %FOOOOOOOOOO
            genThetas(length(Data.theta)-Data.Size+1:end,j)=duty_ratio(:); %randomizer the duty ratios for the simulation
        end
        Data.thetas(:,:,i)=genThetas;
        Data.samples(:,:,:,i)=Data.genModel.simulateImage(genThetas);
    end
    %Compute the accuarcy for each model
    for model_idx=1:Data.nModels
        model=Data.Models{model_idx};
        nParams=model.nParams;
        Data.CRLB{model_idx}=zeros(nParams,Data.nIs);
        for method_idx=1:Data.nMethods
            method=Data.Methods{method_idx};
            Data.bias{model_idx,method_idx}=zeros(model.nParams, Data.nIs);
            Data.var{model_idx,method_idx}=zeros(model.nParams, Data.nIs);
            Data.mean_est_var{model_idx,method_idx}=zeros(model.nParams, Data.nIs);
            Data.rmse{model_idx,method_idx}=zeros(model.nParams, Data.nIs);
            Data.mean_llh{model_idx,method_idx}=zeros(1, Data.nIs);
            for i=1:Data.nIs
                thetas=Data.thetas(1:nParams,:,i);
                samples=Data.samples(:,:,:,i);
                obsCRLB=model.CRLB(thetas);
                
                obsCRLB(:,any( obsCRLB<0 | ~isfinite(obsCRLB),2))=[];
                Data.CRLB{model_idx}(:,i)=mean(obsCRLB,2);
                fprintf('Model(%i/%i)(%s) Method(%i/%i)(%s) Intensity: %f (%i/%i)\n',model_idx, ...
                        Data.nModels, model.Name ,method_idx,Data.nMethods, method, Data.Is(i),i,Data.nIs);
                method_parts=strsplit(method);
                if length(method_parts)==2
                    [theta_est,post_cov]=model.estimatePosterior(samples,str2double(method_parts{2}));
                    llh_est=model.LLH(samples,theta_est);
                    mean_post_var=zeros(model.nParams,1);
                    for k=1:Data.Count
                        mean_post_var=mean_post_var+diag(post_cov(:,:,k));
                    end
                    mean_post_var=mean_post_var/Data.Count;
                    Data.mean_est_var{model_idx,method_idx}(:,i)=mean_post_var;
                else
                    [theta_est, crlb_est, llh_est, ~]=model.estimateMAP(samples,method);
                    
                    crlb_est(:,any( crlb_est>1e6 | crlb_est<0 | ~isfinite(crlb_est),2))=[];
                    Data.mean_est_var{model_idx,method_idx}(:,i)=mean(crlb_est,2);
                end
%                 Data.mean_est_err{model_idx,method_idx}(:,i)=model.estimationAccuracy(theta_est,method);
                error=thetas-theta_est;
                Data.bias{model_idx,method_idx}(:,i)=mean(error,2);
                Data.var{model_idx,method_idx}(:,i)=var(theta_est,[],2);
                Data.rmse{model_idx,method_idx}(:,i)=sqrt(mean(error.*error,2));
                Data.mean_llh{model_idx,method_idx}(i)=mean(llh_est);
            end
        end
    end
end
