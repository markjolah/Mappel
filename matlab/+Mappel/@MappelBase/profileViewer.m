
function fig = profileViewer(obj, theta, im, param_idx)
    if nargin<3 || isempty(im)
        im = obj.simulateImage(theta);
    end
    if nargin<4
        param_idx=1;
    end
%     methods={'SimulatedAnnealing','NewtonDiagonal','Newton','TrustRegion'};
    methods={'Newton','TrustRegion'};
    colors.NewtonDiagonal=[.5,.5,0];
    colors.QuasiNewton=[0,0,1];
    colors.Newton=[0,.5,.5];
    colors.TrustRegion=[0,0,0];
    colors.SimulatedAnnealing=[1,0,1];
    
    colors.prof=[.5 .25 1];
    colors.prof_contour=[.75 .5 1];
    colors.prof_bnd=colors.prof;
    colors.exp=[0 .65 .25];
    colors.exp_bnd=colors.exp;
    colors.obs=[1 .25 .25];
    colors.obs_bnd=colors.obs;
        
    colors.mle = [1 0 0];
    colors.mle_edge = [.5 0 0];
    colors.theta = [0 0 1];
    colors.theta_edge = [0 0 .5];
    colors.init = [0 .5 1];
    colors.init_edge = [0 .25 .5];
    colors.bg = .05*ones(1,3);

    bnd_sty=':';
    lg_fs=8;
    ms=5;
    confidence=0.95;
    N=50;
    epsilon = 1e-7;
    xs_scale = 10; %scale factor for upper bound of unbounded paramters.
    
    
    c = -chi2inv(confidence,1)/2;
    theta_rllh = obj.modelRLLH(im,theta);
    theta_init = obj.estimate(im,'Heuristic');
    init_rllh = obj.modelRLLH(im,theta_init);
    [theta_mle,mle_rllh,obsI,mle_stats] = obj.estimate(im,'TrustRegion',theta_init);
    fisherI = obj.expectedInformation(theta_mle);
    [exp_lb,exp_ub]=obj.errorBoundsExpected(theta_mle,confidence);
    [obs_lb,obs_ub]=obj.errorBoundsObserved(im,theta_mle,confidence,obsI);
    
    [prof_lb, prof_ub, prof_pts_lb, prof_pts_ub, prof_rllh_lb, prof_rllh_ub] = ...
        obj.errorBoundsProfileLikelihood(im,theta_mle,confidence,mle_rllh,obsI,'Newton');
    
    
    if isfinite(obj.ParamUBound(param_idx))
        xs_lim = [obj.ParamLBound(param_idx)+epsilon, obj.ParamUBound(param_idx)-epsilon];
    else
        xs_lim = [obj.ParamLBound(param_idx)+epsilon,xs_scale*theta_mle(param_idx)];
    end
    xs=linspace(xs_lim(1),xs_lim(2),N);
    
    
    theta_mle_slice=repmat(theta_mle,1,N);
    theta_mle_slice(param_idx,:)=xs;
    theta_mle_slice_rllh = obj.modelRLLH(im,theta_mle_slice) - mle_rllh;
    
    theta_true_slice=repmat(theta(:),1,N);
    theta_true_slice(param_idx,:)=xs;
    theta_true_slice_rllh = obj.modelRLLH(im,theta_true_slice) - mle_rllh;
    
    theta_init_slice=repmat(theta_init(:),1,N);
    theta_init_slice(param_idx,:)=xs;
    theta_init_slice_rllh = obj.modelRLLH(im,theta_init_slice) - mle_rllh;
    
    
    fixed_params=zeros(obj.NumParams,1,'uint64');
    fixed_params(param_idx)=1;
    for method_cell=methods
        method = method_cell{1};
        [prof_rllh.(method), prof_params.(method), stats.(method)]=obj.estimateProfileLikelihood(im,fixed_params,xs,method,theta_mle);
        prof_rllh.(method)=prof_rllh.(method)-mle_rllh;
    end

    inbounds = prof_rllh.('TrustRegion')>2*c;
    inbounds(max(1,find(inbounds,1,'first')-1))=1;
    inbounds(min(numel(inbounds),find(inbounds,1,'last')+1))=1;
    zoom_xs = xs(inbounds);
    if isempty(zoom_xs)
        zoom_xs=xs;
    end
    zoom_xs_lim = [min([zoom_xs';exp_lb(param_idx);obs_lb(param_idx);prof_lb(param_idx)]),max([zoom_xs';exp_ub(param_idx);obs_ub(param_idx);prof_ub(param_idx)])];
    zoom_xs_lim = [max(obj.ParamLBound(param_idx)+epsilon,zoom_xs_lim(1)), min(obj.ParamUBound(param_idx)-epsilon,zoom_xs_lim(2))];
    zoom_xs = linspace(zoom_xs_lim(1),zoom_xs_lim(2),N);
    zoom_theta_mle_slice = repmat(theta_mle,1,N);
    zoom_theta_mle_slice(param_idx,:) = zoom_xs;
    zoom_theta_mle_slice_rllh = obj.modelRLLH(im,zoom_theta_mle_slice) - mle_rllh;
    zoom_theta_true_slice=repmat(theta(:),1,N);
    zoom_theta_true_slice(param_idx,:)=zoom_xs;
    zoom_theta_true_slice_rllh = obj.modelRLLH(im,zoom_theta_true_slice) - mle_rllh;
    for method_cell=methods
        method = method_cell{1};
        [zoom_prof_rllh.(method),zoom_prof_params.(method),zoom_stats.(method)]=obj.estimateProfileLikelihood(im,fixed_params,zoom_xs,method,theta_mle);
        zoom_prof_rllh.(method)=zoom_prof_rllh.(method)-mle_rllh;
    end
 
    %Normalize so that theta_mle=0;
    init_rllh=init_rllh-mle_rllh;
    theta_rllh=theta_rllh-mle_rllh;
    prof_rllh_lb = prof_rllh_lb - mle_rllh;
    prof_rllh_ub = prof_rllh_ub - mle_rllh;
    
    dx = xs-theta_mle(param_idx);
    obs_llh_model=-.5*dx.^2*obsI(param_idx,param_idx);
    
    zoom_dx = zoom_xs-theta_mle(param_idx);
    obsIinv = pinv(obsI);
    zoom_obs_llh_model=-.5*zoom_dx.^2*1./obsIinv(param_idx,param_idx);
    fisherIinv = pinv(fisherI);
    zoom_expected_llh_model=-.5*zoom_dx.^2*1./fisherIinv(param_idx,param_idx);
    
    
    fixed_params2d=zeros(obj.NumParams,1,'uint64');
    fixed_params2d(1)=1;
    fixed_params2d(2)=1;
    delta = prof_ub-prof_lb;
    xs2d = linspace(max(obj.ParamLBound(1)+epsilon,prof_lb(1)-delta(1)*0.1), min(obj.ParamUBound(1)-epsilon,prof_ub(1)+delta(1)*0.1), N);
    ys2d = linspace(max(obj.ParamLBound(2)+epsilon,prof_lb(2)-delta(2)*0.1), min(obj.ParamUBound(2)-epsilon,prof_ub(2)+delta(2)*0.1), N);
    [X, Y] = meshgrid(xs2d,ys2d);
    theta_mle_slice2d = repmat(theta_mle,1,N*N);
    theta_mle_slice2d(1,:)=X(:);
    theta_mle_slice2d(2,:)=Y(:);
    [prof2d_rllh,~] = obj.estimateProfileLikelihood(im,fixed_params2d,theta_mle_slice2d(1:2,:),'trustregion',theta_mle);
    prof2d_rllh = prof2d_rllh-mle_rllh;
    Z = reshape(prof2d_rllh,N,N);
    
%     fisherIinv = pinv(fisherI);
%     zoom_dx = zoom_xs-theta_mle(param_idx);
%     zoom_profile_llh_model=-.5*zoom_dx.^2*(1./fisherIinv(param_idx,param_idx));
    
    fig = figure('Position',[10,10,1620,1000]);
%     whitebg();
    ax=axes('Position',[.05,.55,.33,.4]);
    hold('on')
    plot(theta(param_idx),theta_rllh,'s','MarkerEdgeColor',colors.theta_edge,'MarkerFaceColor',colors.theta,'MarkerSize',5,'DisplayName','True theta');
    plot(xs,theta_true_slice_rllh,'-','Color',colors.theta,'LineWidth',1,'DisplayName','True theta LLH slice')
    
    plot(theta_mle(param_idx),0,'o','MarkerEdgeColor',colors.mle_edge,'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName','Theta MLE');
    plot(xs,theta_mle_slice_rllh,'-','Color',colors.mle,'LineWidth',2,'DisplayName','Theta MLE LLH slice')
    
    plot(theta_init(param_idx),init_rllh,'^','MarkerEdgeColor',colors.init_edge,'MarkerFaceColor',colors.init,'MarkerSize',ms,'DisplayName','Theta init');
    plot(xs,theta_init_slice_rllh,'-','Color',colors.init,'LineWidth',1,'DisplayName','Theta Init LLH sice')
    
    plot(xs,obs_llh_model,'-.','Color',colors.obs,'LineWidth',1,'DisplayName','Observed info Model')
    for method_cell=methods
        method = method_cell{1};
        plot(xs,prof_rllh.(method),'-','Color',colors.(method),'DisplayName',sprintf('Profile LLH (%s)',method))
    end
    plot(xs,c*ones(N,1),':','Color',[0,0,0],'DisplayName','Chi-sq LLH ratio')
    lg=legend('location','best');
    lg.Box='off';
    lg.FontSize=lg_fs;
    ylabel('relative log-likelihood')
    xlabel(obj.ParamNames{param_idx})
    ylim([min([init_rllh;prof_rllh.('TrustRegion');theta_mle_slice_rllh;theta_rllh;theta_true_slice_rllh;theta_init_slice_rllh;]),-.25*c])
    xlim([0,xs_lim(2)]);
    title(sprintf('''%s'' Profile Likelihood (full range)',obj.ParamNames{param_idx}));

    ax=axes('Position',[.05,.05,.33,.4]);
    hold('on')
    ylim([min([zoom_prof_rllh.('TrustRegion');zoom_theta_mle_slice_rllh;theta_rllh;zoom_theta_true_slice_rllh]),-.25*c])
    yl=ylim();
    xlim(zoom_xs_lim);
    plot(theta(param_idx),theta_rllh,'s','MarkerEdgeColor',colors.theta_edge,'MarkerFaceColor',colors.theta,'MarkerSize',5,'DisplayName','True theta');
    plot(zoom_xs,zoom_theta_true_slice_rllh,'-','Color',colors.theta,'LineWidth',1,'DisplayName','True theta LLH slice')

    plot(theta_mle(param_idx),0,'o','MarkerEdgeColor',colors.mle_edge, 'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName','Theta MLE');
    plot(zoom_xs,zoom_theta_mle_slice_rllh,'-','Color',colors.mle,'LineWidth',2,'DisplayName','MLE LLH slice')
    
    plot([obs_lb(param_idx),obs_ub(param_idx)],c*[1,1],'s','Color',colors.obs,'DisplayName','Observed Bounds')
    
    plot(zoom_xs,zoom_obs_llh_model,'-.','Color',colors.obs,'LineWidth',.5,'DisplayName','Observed info model')

    plot([exp_lb(param_idx),exp_ub(param_idx)],c*[1,1],'v','Color',colors.exp,'DisplayName','Expected Bounds')
    plot(zoom_xs,zoom_expected_llh_model,'-.','Color',colors.exp,'LineWidth',.5,'DisplayName','Expected info model')

    %plot(zoom_xs,zoom_profile_llh_model,'-.','Color',[.7 .3 .0],'LineWidth',1,'DisplayName','mle profile information model')
    plot([prof_lb(param_idx),prof_ub(param_idx)],[prof_rllh_lb(param_idx),prof_rllh_ub(param_idx)],'o','Color',colors.prof, 'Linewidth',1, 'DisplayName','Profile likelihood bounds')
    for method_cell=methods
        method = method_cell{1};
        plot(zoom_xs,zoom_prof_rllh.(method),'-','Color',colors.(method),'DisplayName',sprintf('Profile LLH (%s)',method))
    end
    plot(zoom_xs,c*ones(N,1),':','Color',[0,0,0],'DisplayName','Chi-sq LLH ratio')
    
    h=plot([obs_lb(param_idx),obs_lb(param_idx)],yl,bnd_sty,'Color',colors.obs_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
    h=plot([obs_ub(param_idx),obs_ub(param_idx)],yl,bnd_sty,'Color',colors.obs_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
    
    h=plot([exp_lb(param_idx),exp_lb(param_idx)],yl,bnd_sty,'Color',colors.exp_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
    h=plot([exp_ub(param_idx),exp_ub(param_idx)],yl,bnd_sty,'Color',colors.exp_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
    
    h=plot([prof_lb(param_idx),prof_lb(param_idx)],yl,bnd_sty,'Color',colors.prof_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
    h=plot([prof_ub(param_idx),prof_ub(param_idx)],yl,bnd_sty,'Color',colors.prof_bnd);
    h.Annotation.LegendInformation.IconDisplayStyle='off';
        
    lg=legend('location','best');
    lg.Box='off';
    lg.FontSize=lg_fs;
    lg.NumColumns=2;
    ylabel('relative log-likelihood')
    xlabel(obj.ParamNames{param_idx})
    title(sprintf('''%s'' Profile Likelihood (zoomed)',obj.ParamNames{param_idx}));
    
    
    ax=axes('Position',[.4,.5,.35,.5]);
    GUIBuilder.positionImageAxes(ax,double(obj.ImageSize),ax.Position,[00 00 00 00]);
    colormap(ax,hot());
    colorbar(ax);
    hold('on');
    xlabel(ax,'X (px)');
    ylabel(ax,'Y (px)');
    axis(ax,'tight');
    pbaspect(double([obj.ImageSize(1) obj.ImageSize(2) 1]))
    ax.TickDir='out';
    ax.XTick=0:obj.ParamUBound(1);
    ax.YTick=0:obj.ParamUBound(2);
    ax.YDir='reverse';
    imagesc(ax,[.5,obj.ParamUBound(1)-.5],[.5,obj.ParamUBound(2)-.5],im);
    plot(theta_init(1),theta_init(2),'^','MarkerEdgeColor',colors.init_edge,'MarkerFaceColor',colors.init,'LineWidth',1,'MarkerSize',ms,'DisplayName','theta init llh');
    plot(theta(1),theta(2),'s','LineWidth',1,'MarkerEdgeColor',colors.theta_edge,'MarkerFaceColor',colors.theta,'MarkerSize',7,'DisplayName','true theta');
    plot(theta_mle(1),theta_mle(2),'o','LineWidth',1,'MarkerEdgeColor',colors.mle_edge,'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName','theta mle');
    plot([exp_lb(1) exp_ub(1) exp_ub(1) exp_lb(1) exp_lb(1)], [exp_lb(2) exp_lb(2) exp_ub(2) exp_ub(2),exp_lb(2)],'-','Color',colors.exp,'LineWidth',1,'DisplayName','Expected bounds');
    plot([obs_lb(1) obs_ub(1) obs_ub(1) obs_lb(1) obs_lb(1)], [obs_lb(2) obs_lb(2) obs_ub(2) obs_ub(2),obs_lb(2)],'-','Color',colors.obs,'LineWidth',1,'DisplayName','Observed bounds');
    plot([prof_lb(1) prof_ub(1) prof_ub(1) prof_lb(1) prof_lb(1)], [prof_lb(2) prof_lb(2) prof_ub(2) prof_ub(2),prof_lb(2)],'-','Color',colors.prof,'LineWidth',1,'DisplayName','Profile bounds');
    contour(X,Y,Z,[c c],'Color',colors.prof_contour,'LineWidth',2,'DisplayName','Profile bounds contour');
    lg=legend('location','best');
    lg.FontSize=lg_fs;
    lg.Box='on';
    title('Image')
    
    ax=axes('Position',[.7,.5,.35,.5]);
    GUIBuilder.positionImageAxes(ax,double(obj.ImageSize),ax.Position,[00 0 0 0]);
    colormap(ax,hot());
%     colorbar(ax);
    hold('on');
    xlabel(ax,'X (px)');
    ylabel(ax,'Y (px)');
    pbaspect(double([obj.ImageSize(1) obj.ImageSize(2) 1]))
    bds=[exp_lb(1:2),obs_lb(1:2),prof_lb(1:2),exp_ub(1:2),obs_ub(1:2),prof_ub(1:2)];
    min_bds = min(bds,[],2);
    max_bds = max(bds,[],2);
    rng_bds= max_bds-min_bds;
    rng_bds_hw = max(rng_bds)/2;
    rng_bds_hw = min(2*rng_bds_hw,rng_bds_hw+1);
    ctr_bds = .5*(max_bds+min_bds);
    lims = [ctr_bds-repmat(rng_bds_hw,2,1), ctr_bds+repmat(rng_bds_hw,2,1)];
    lb_over= lims(:,1)-obj.ParamLBound(1:2);
    lb_over(lb_over>0)=0;
    lims= lims - repmat(lb_over,1,2);
    ub_over= obj.ParamUBound(1:2)-lims(:,2);
    ub_over(ub_over>0)=0;
    lims = lims + repmat(ub_over,1,2);

    axis(ax,'tight');
    ax.TickDir='out';
    ax.YDir='reverse';
%     ax.GridLineStyle=':';
    
    imagesc(ax,[.5,obj.ParamUBound(1)-.5],[.5,obj.ParamUBound(2)-.5],im,'AlphaData',0.3);
    xlim(lims(1,:));
    ylim(lims(2,:));
%     ax.XTick=linspace(lims(1,1),lims(1,2),6);
%     ax.YTick=linspace(lims(2,1),lims(2,2),6);
    grid('on');
    grid('minor');
    plot(theta_init(1),theta_init(2),'^','MarkerEdgeColor',colors.init_edge,'MarkerFaceColor',colors.init,'LineWidth',1,'MarkerSize',ms,'DisplayName','theta init llh');
    plot(theta(1),theta(2),'s','LineWidth',1,'MarkerEdgeColor',colors.theta_edge,'MarkerFaceColor',colors.theta,'MarkerSize',7,'DisplayName','true theta');
    plot(theta_mle(1),theta_mle(2),'o','LineWidth',1,'MarkerEdgeColor',colors.mle_edge,'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName','theta mle');
    plot([exp_lb(1) exp_ub(1) exp_ub(1) exp_lb(1) exp_lb(1)], [exp_lb(2) exp_lb(2) exp_ub(2) exp_ub(2),exp_lb(2)],'-','Color',colors.exp,'LineWidth',1,'DisplayName','Expected bounds');
    plot([obs_lb(1) obs_ub(1) obs_ub(1) obs_lb(1) obs_lb(1)], [obs_lb(2) obs_lb(2) obs_ub(2) obs_ub(2),obs_lb(2)],'-','Color',colors.obs,'LineWidth',1,'DisplayName','Observed bounds');
    plot([prof_lb(1) prof_ub(1) prof_ub(1) prof_lb(1) prof_lb(1)], [prof_lb(2) prof_lb(2) prof_ub(2) prof_ub(2),prof_lb(2)],'-','Color',colors.prof,'LineWidth',1,'DisplayName','Profile bounds');
    contour(X,Y,Z,[c c],'Color',colors.prof_contour,'LineWidth',2,'DisplayName','Profile bounds contour');
%     lg=legend('location','best');
%     lg.FontSize=lg_fs;
%     lg.Box='on';

    title('Image (zoomed)')
    
    
    plot_pos=[.45 .32 .20 .17;
              .75 .32 .20 .17;
              .45 .06 .20 .17;
              .75 .06 .20 .17];
    nplots=1;
    for idx = 1:obj.NumParams
        if idx == param_idx; continue; end
        ax = axes('Position',plot_pos(nplots,:));
        nplots=nplots+1;
        hold('on');
        for method_cell=methods
            method = method_cell{1};
            thetas = prof_params.(method);
            plot(xs,thetas(idx,:),'-','Color',colors.(method),'DisplayName',sprintf('Prof. (%s)',method))
        end
        xlabel(obj.ParamNames(param_idx));
        title(obj.ParamNames(idx));
        plot(theta_mle(param_idx),theta_mle(idx),'o','MarkerEdgeColor',colors.mle_edge,'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName',sprintf('theta mle llh%.9g',mle_rllh));
    end
end
