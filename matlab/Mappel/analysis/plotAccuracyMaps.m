function plotAccuracyMaps(Data,projection)
    f=figure();
    colormap(jet());
    nModels=length(Data.model);
    nEstimators=length(Data.estimator);
%     plotBlinkPattern= ~all(logical(Data.blinkpattern));
    plotBlinkPattern= true;
        
    if isfield(Data,'blinkpattern')
        nModels=nModels+1;
    end
    subplot(111);
    maps=cell(nModels-1,nEstimators);
    maxv=0;
    minv=0;
    for mi=1:length(Data.model)
        nParams=Data.model{mi}.nParams;
        for ei=1:nEstimators
%             projection=[2];
            if any(projection>nParams) 
                maps{mi,ei}=zeros(Data.gridsize);
                continue;
            end
            etheta=Data.theta_est_grid{mi,ei};
            theta=Data.theta_grid(1:nParams,:,:,:);
            evar=Data.est_var_grid{mi,ei};
            err=squeeze(mean(theta-etheta,2));            
            maps{mi,ei}=abs(squeeze(err(projection,:,:)));
            minv=min(minv,min(maps{mi,ei}(:)));
            maxv=max(maxv,max(maps{mi,ei}(:)));
        end
    end
    for mi=1:length(Data.model)
        for ei=1:nEstimators
            subplot(nModels, nEstimators, (mi-1)*(nEstimators)+ei);
%             subplot(nModels, nEstimators+1, (mi-1)*(nEstimators+1)+ei);
            Data.model{mi}.plotAccuracyMap(maps{mi,ei});
            if ei==1
                ylabel(sprintf('%s',Data.model{mi}.Name));
            end
            if mi==length(Data.model)
                xlabel(sprintf('%s',Data.estimator{ei}));
            end
            caxis([minv maxv]);
            colorbar();
        end
%         subplot(nModels, nEstimators+1,(mi)*(nEstimators+1));
%         evar=squeeze(mean(Data.est_var_grid{mi,ei}(1,);
%         Data.model{mi}.plotAccuracyMap(Data.crlb_grid{mi,1}');
%         if mi==length(Data.model)
%             xlabel('CRLB');
%         end
%         caxis([0 maxv]);
%         colorbar();


    end
    if isfield(Data,'blinkpattern')
        subplot(nModels, 1,nModels);
        bar(0:length(Data.blinkpattern)-1,Data.blinkpattern,1)
        title(sprintf('I:%g bg:%g sigma:%g',Data.theta(3),Data.theta(4),Data.theta(5)));
        xlabel('Column (x) (pixels)');
        xlim([0.5,length(Data.blinkpattern)-0.5]);
        ylabel('Duty ratio');
        colorbar();
    end
end
