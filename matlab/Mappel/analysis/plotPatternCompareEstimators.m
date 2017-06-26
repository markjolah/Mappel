function plotPatternCompareEstimators(Data, param)
    if nargin==1
        param=1;
    end
    lw=2;

    %RMSE analysis
    figure('Position',[1 1 1200 500]);
    subplotAX=cell(1,Data.nModels);
    maxRMSE=0;
    for model_idx=1:Data.nModels
        model=Data.Models{model_idx};
        pname=model.ParamNames{param};
        punits=model.ParamUnits{param};
        subplotAX{model_idx}=subplot(1,Data.nModels,model_idx);
        hold('on');
        colormap(lines(Data.nMethods));
        cmap=colormap();
        for method_idx=1:Data.nMethods
            method=Data.Methods{method_idx};
            color=cmap(method_idx,:);
            disp_name=sprintf('%s $(\\theta_{%s})$',method,pname);
            rmse=Data.rmse{model_idx,method_idx}(param,:);
            plot(Data.Is, rmse,'-','Color',color,'LineWidth', lw,'DisplayName',disp_name);
            maxRMSE=max(maxRMSE,max(rmse(:)));
            est_rmse=sqrt(Data.mean_est_var{model_idx,method_idx}(param,:));
            if method_idx==Data.nMethods
                disp_name=sprintf('Est. RMSE %s $(\\theta_{%s})$',method,pname);
                plot(Data.Is, est_rmse,':','Color',color,'LineWidth', lw,'DisplayName',disp_name);
            end
        end
        set(gca(),'YGrid','on','XGrid','on');
        disp_name=sprintf('$\\sqrt{\\mathrm{CRLB}(\\theta_%s)}$',pname);
        plot(Data.Is, sqrt(Data.CRLB{model_idx}(param,:)),'-k','DisplayName',disp_name);
        hold('off');
        lh=legend('Location','NorthEast');
        set(lh,'interpreter','latex');
        title(sprintf('Model:%s -- RMSE$(\\theta_{%s})$ Comparison',model.Name,pname),'interpreter','latex');
        xlabel('Intensity [photons]','interpreter','latex');
        ylabel(sprintf('RMSE$(\\theta_{%s})$ [%s]',pname,punits),'interpreter','latex');
    end
    for model_idx=1:Data.nModels
        set(subplotAX{model_idx},'YLim',[0,min(1,maxRMSE)]);
    end

    %Bias analysis
    figure('Position',[50 50 1200 500]);
    subplotAX=cell(1,Data.nModels);
    maxBias=0;
    minBias=0;
    for model_idx=1:Data.nModels
        model=Data.Models{model_idx};
        pname=model.ParamNames{param};
        punits=model.ParamUnits{param};
        subplotAX{model_idx}=subplot(1,Data.nModels,model_idx);
        hold('on');
        for method_idx=1:Data.nMethods
            method=Data.Methods{method_idx};
            disp_name=sprintf('%s $(\\theta_{%s})$',method,pname);
            bias=Data.bias{model_idx,method_idx}(param,:);
            plot(Data.Is, bias,'-','LineWidth', lw,'DisplayName',disp_name);
            minBias=min(minBias,min(bias(:)));
            maxBias=max(maxBias,max(bias(:)));
        end
        plot([0,Data.maxI], [0 0] ,'-k','DisplayName','Unbiased');
        hold('off');
        lh=legend('Location','NorthEast');
        set(lh,'interpreter','latex');
        title(sprintf('Model:%s -- Bias$(\\theta_{%s})$ Comparison',model.Name,pname),'interpreter','latex');
        xlabel('Intensity [photons]','interpreter','latex');
        ylabel(sprintf('Bias$(\\theta_{%s})$ [%s]',pname,punits),'interpreter','latex');
    end
    for model_idx=1:Data.nModels
        set(subplotAX{model_idx},'YLim',[minBias,maxBias]);
    end

    %Variance analysis
    figure('Position',[100 100 1200 500]);
    subplotAX=cell(1,Data.nModels);
    maxVar=0;
    for model_idx=1:Data.nModels
        model=Data.Models{model_idx};
        pname=model.ParamNames{param};
        punits=model.ParamUnits{param};
        subplotAX{model_idx}=subplot(1,Data.nModels,model_idx);
        hold('on');
        colormap('lines');
        cmap=colormap();
        for method_idx=1:Data.nMethods
            method=Data.Methods{method_idx};
            color=cmap(method_idx,:);
            Var=Data.var{model_idx,method_idx}(param,:);
            VarEst=Data.mean_est_var{model_idx,method_idx}(param,:);
            disp_name=sprintf('Var(%s) $(\\theta_{%s})$',method,pname);
            plot(Data.Is, Var,'-','Color',color,'LineWidth', lw,'DisplayName',disp_name);
            maxVar=max(maxVar,max(Var(:)));
            disp_name=sprintf('Est. Var(%s) $(\\theta_{%s})$',method,pname);
            plot(Data.Is, VarEst,':','Color',color,'LineWidth', lw,'DisplayName',disp_name);
            maxVar=max(maxVar,max(VarEst(:)));
        end
        disp_name=sprintf('$\\mathrm{CRLB}(\\theta_%s)$',pname);
        plot(Data.Is, Data.CRLB{model_idx}(param,:),'-k','DisplayName',disp_name);
        hold('off');
        lh=legend('Location','NorthEast');
        set(lh,'interpreter','latex');
        title(sprintf('Model:%s -- Variance$(\\theta_{%s})$ Comparison',model.Name,pname),'interpreter','latex');
        xlabel('Intensity [photons]','interpreter','latex');
        ylabel(sprintf('Variance$(\\theta_{%s})$ [%s]',pname,punits),'interpreter','latex');
    end
    if maxVar>1e4
        maxVar=2;
    end
    for model_idx=1:Data.nModels       
        set(subplotAX{model_idx},'YLim',[0,maxVar]);
    end



end


