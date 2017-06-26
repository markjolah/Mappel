function plotPointCompareEstimators(Data, params)
    if nargin==1
        params=[1];
    end
    
    %RMSE analysis
    f=figure();
    hold on;
    cmap=lines(Data.nMethods+1);
    cmap(2,:)=[];
    ms=7;
    lw=2;
    lss={'-','--',':'};
    pnames=Data.model.ParamNames(params);
    ps=zeros(length(params),Data.nMethods); %line series handles
    crlb_ps=zeros(length(params),1);
    for m=1:Data.nMethods
        name=Data.Methods{m};
        fprintf('Method(%i) Name: %s\n',m,name);
        for p=1:length(params)
            disp_name=sprintf('%s (%s)',name,pnames{p});
            ps(p,m)=plot(Data.Is, Data.rmse{m}(params(p),:),lss{p}, 'Color', cmap(m,:),'MarkerEdgeColor',cmap(m,:),...
                 'MarkerFaceColor',cmap(m,:),'MarkerSize', ms, 'LineWidth', lw,'DisplayName',disp_name );
        end
    end
    for p=1:length(params)
        disp_name=sprintf('$\\sqrt{\\mathrm{CRLB}(\\theta_%s)}$',pnames{p});
        crlb_ps(p)=plot(Data.Is, sqrt(Data.CRLB(params(p),:)), 'LineStyle', lss{p}, 'Color','k',...
                            'DisplayName',disp_name);
    end
    set(gca(),'Ylim',[0 1]);
    set(gca(),'YGrid','on','XGrid','on','YMinorGrid','on','YminorTick','on','XminorTick','on');
    % t=sprintf('(\\theta_x,\\theta_y)=(%.4g,%.4g),\\quad \\theta_{\\mathrm{bg}}:%.3g),\\quad \\theta_{\\mathrm{\\sigma}}:%.3g',Data.Theta(1),Data.Theta(2),Data.Theta(4),Data.Theta(5));
    % t=sprintf('(x,y)=(%.4g,%.4g), bg:%.3g, sigma:%.3g',Data.Theta(1),Data.Theta(2),Data.Theta(4),Data.Theta(5));
    % title(['$' t '$'],'interpreter','latex')
%     title('RMSE Comparison');
    xlabel('Intensity [photons]','interpreter','latex','FontSize',14);
    ylabel(sprintf('RMSE$(\\theta_{%s})$ [%s]',pnames{1},Data.model.ParamUnits{params}),'interpreter','latex','FontSize',12);
    lh=legend([ps(:)' crlb_ps],'Location','NorthEast');
    set(lh,'interpreter','latex','FontSize',14);
    hold off;

    %Bias Analysis
    f=figure();
    hold('on');
    for m=1:Data.nMethods
        name=Data.Methods{m};
        fprintf('Method(%i) Name: %s\n',m,name);
        for p=1:length(params)
            disp_name=sprintf('%s (%s)',name,pnames{p});
            ps(p,m)=plot(Data.Is, Data.bias{m}(params(p),:),lss{p}, 'Color', cmap(m,:),'MarkerEdgeColor',cmap(m,:),...
                 'MarkerFaceColor',cmap(m,:),'MarkerSize', ms, 'LineWidth', lw,'DisplayName',disp_name );
        end
    end
    feducial_h=plot(Data.Is, zeros(size(Data.Is)), '-k','DisplayName','Unbiased');
    title('Bias Comparison');
    xlabel('Intensity (photons)');
    ylabel('Bias (mean error) (pixels)');
    lh=legend([ps(:); feducial_h],'Location','NorthEast');
    set(lh,'interpreter','latex');
    hold('off');
    
    %Variance analysis
    f=figure();
    hold('on');
    var_ps=zeros(length(params),Data.nMethods); %line series handles
    for m=1:Data.nMethods
        name=Data.Methods{m};
        fprintf('Method(%i) Name: %s\n',m,name);
        for p=1:length(params)
            disp_name=sprintf('%s Var(%s)',name,pnames{p});
            ps(p,m)=plot(Data.Is, Data.var{m}(params(p),:),lss{p}, 'Color', cmap(m,:),'MarkerEdgeColor',cmap(m,:),...
                 'MarkerFaceColor',cmap(m,:),'MarkerSize', ms, 'LineWidth', lw,'DisplayName',disp_name );
            disp_name=sprintf('%s Est. Var(%s)',name,pnames{p});
            var_ps(p,m)=plot(Data.Is, Data.mean_est_var{m}(params(p),:), 'LineStyle', ':', 'Color',cmap(m,:),'LineWidth',lw,...
                            'DisplayName',disp_name);
        end
    end
    for p=1:length(params)
        disp_name=sprintf('$\\sqrt{\\mathrm{CRLB}(\\theta_%s)}$',pnames{p});
        crlb_ps(p)=plot(Data.Is, Data.CRLB(params(p),:), 'LineStyle', lss{p}, 'Color','k',...
                            'DisplayName',disp_name);
    end
    title('Variance Comparison');
    xlabel('Intensity (photons)');
    ylabel('Variance (mean error) (pixels^2)');
    ylim([0 2]);
    lh=legend([ps(:)' var_ps(:)' crlb_ps],'Location','NorthEast');
    set(lh,'interpreter','latex');
    hold('off');

    test_I=750;
    test_theta=Data.Theta;
    test_theta(3)=test_I;
    test_theta=[3.8000    5.4000  671.0000    5.0000    1.3000    1.0000    0.93  0.66 0.69 0.13 1 0.64 0.35];
%       test_theta(6:enmd)=Data.qdsim.simulateImage();
    f=figure();
    clf(f);
    Data.genModel.viewDipImage(Data.genModel.modelImage(test_theta),f);
    hold on;
    plot( Data.Theta(1)-0.5, Data.Theta(2)-0.5, 'p','markerfacecolor',[1.0,0.1,1.0],'markeredgecolor','m','linewidth',2,'MarkerSize',15);
    hold off;
    set(f,'Name',sprintf('Model Image (I=%g)',test_theta(3)));

    f=figure();
    clf(f);
    Data.genModel.viewDipImage(Data.genModel.simulateImage(test_theta),f);
    hold on;
    plot( Data.Theta(1)-0.5, Data.Theta(2)-0.5, 'p','markerfacecolor',[1.0,0.1,1.0],'markeredgecolor','m','linewidth',2,'MarkerSize',15);
    hold off;
    set(f,'Name',sprintf('Simulated Image (I=%g)',test_theta(3)));
end
