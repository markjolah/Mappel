
function fig = profileViewer2D(obj, theta, im, method, fixedI)
    if nargin<2
        theta = obj.samplePrior();
    else
        theta = obj.boundTheta(theta);
    end
    if nargin<3 || isempty(im)
        im = obj.simulateImage(theta);
    end
    if nargin<4
        method='TrustRegion';
    end
    if nargin<5
        fixedI=[];
    end
    colors.prof=[.5 .25 1];
    colors.prof_contour=[.75 .5 1];
    colors.prof_bnd=colors.prof;
    colors.mle = [1 0 0];
    colors.mle_edge = [.5 0 0];
    colors.theta = [0 0 1];
    colors.theta_edge = [0 0 .5];
    lg_fs=8;
    ms=5;

    confidence=0.95;
    c=-chi2inv(confidence,1)/2;
    theta_rllh = obj.modelRLLH(im,theta);
    theta_init = obj.estimateMax(im,'Heuristic');
    init_rllh = obj.modelRLLH(im,theta_init);
    [theta_mle,mle_rllh,obsI,mle_stats] = obj.estimateMax(im,method,theta_init);
    crlb = obj.CRLB(theta);
%     [exp_lb,exp_ub]=obj.errorBoundsExpected(theta_mle,confidence);
%     [obs_lb,obs_ub]=obj.errorBoundsObserved(im,theta_mle,confidence,obsI);
%     [min_lb,max_ub]=obj.errorBoundsExpected(theta_mle,0.999);
    N=100;
%     xs=linspace(0.6*min_lb(param_idx),1.5*max_ub(param_idx),N);
    xs = linspace(0.1,7.9,N);
    ys = linspace(0.1,7.9,N);
    [X, Y] = meshgrid(xs,ys);
    aspect = [double(obj.ImageSize(2))/double(obj.ImageSize(1)),1,0.8];
    marg_thetas = repmat(theta_mle,1,N*N);
    marg_thetas(1,:)=X(:);
    marg_thetas(2,:)=Y(:);
    if ~isempty(fixedI)
        marg_thetas(3,:)=fixedI;
        [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,1:3,marg_thetas(1:3,:),method,theta);
    else
        [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,1:2,marg_thetas(1:2,:),method,theta);
    end
    
    [prof_lb, prof_ub, prof_pts_lb, prof_pts_ub, prof_rllh_lb, prof_rllh_ub] = ...
        obj.errorBoundsProfileLikelihood(im,theta_mle,confidence,mle_rllh,obsI,'Newton');
%     marg_rllh = obj.modelRLLH(im,marg_thetas);
    
%     fixed_params=zeros(obj.NumParams,1,'uint64');
%     fixed_params(1)=1;
%     fixed_params(2)=1;

    prof_rllh = prof_rllh-mle_rllh;
    
    Z=reshape(prof_rllh,N,N);
    figH=figure('Position',[10,10,1000,700]);
    whitebg(figH);
    ax=subplot(221);
    GUIBuilder.positionImageAxes(ax,double(obj.ImageSize),ax.Position,[00 00 00 00]);
    imagesc(ax,[.5,obj.ParamUBound(1)-.5],[.5,obj.ParamUBound(2)-.5],im);
    hold('on');
    plot(theta(1),theta(2),'s','LineWidth',1,'MarkerEdgeColor',colors.theta_edge,'MarkerFaceColor',colors.theta,'MarkerSize',7,'DisplayName','true theta');
    plot(theta_mle(1),theta_mle(2),'o','LineWidth',1,'MarkerEdgeColor',colors.mle_edge,'MarkerFaceColor',colors.mle,'MarkerSize',ms,'DisplayName','theta mle');
    plot([prof_lb(1) prof_ub(1) prof_ub(1) prof_lb(1) prof_lb(1)], [prof_lb(2) prof_lb(2) prof_ub(2) prof_ub(2),prof_lb(2)],'-','Color',colors.prof,'LineWidth',1,'DisplayName','Profile bounds');
    contour(X,Y,Z,[c c],'Color',colors.prof_contour,'LineWidth',2,'DisplayName','Profile bounds contour');
    colormap(ax,hot());
    colorbar(ax);
    lg=legend('location','best');
    lg.FontSize=lg_fs;
    lg.Box='on';
    title('Image')
    
    xlabel(ax,'X (px)');
    ylabel(ax,'Y (px)');
    axis(ax,'tight');
    pbaspect(aspect);

    ax=subplot(222);
    hold('on')
    title('llh')
    ax.YDir='reverse';
    ax.TickDir='out';
    ax.XMinorTick='on';
    ax.YMinorTick='on';
    ax.ZMinorTick='on';
    ax.XGrid='on';
    ax.YGrid='on';
    ax.ZGrid='on';
    ax.Box='on';
    ax.BoxStyle='full';
    ax.Projection='Orthographic';
    view(23,13);
    pbaspect(aspect);
    xlabel('x (px)');
    ylabel('y (px)');
    zlabel('llh');
    ax.Color=[0.1,0.1,0.1];
    colormap(ax,'parula')
    contour3(X,Y,Z,30);
    contour3(X,Y,Z,[c c],'LineWidth',2)
    plot3(prof_lb(1),prof_pts_lb(2,1),prof_rllh_lb(1)-mle_rllh,'o','MarkerEdgeColor',colors.prof_bnd,'MarkerFaceColor',colors.prof,'MarkerSize',ms);
    plot3(prof_pts_lb(1,2),prof_lb(2),prof_rllh_lb(2)-mle_rllh,'o','MarkerEdgeColor',colors.prof_bnd,'MarkerFaceColor',colors.prof,'MarkerSize',ms);
    plot3(prof_ub(1),prof_pts_ub(2,1),prof_rllh_ub(1)-mle_rllh,'o','MarkerEdgeColor',colors.prof_bnd,'MarkerFaceColor',colors.prof,'MarkerSize',ms);
    plot3(prof_pts_ub(1,2),prof_ub(2),prof_rllh_ub(2)-mle_rllh,'o','MarkerEdgeColor',colors.prof_bnd,'MarkerFaceColor',colors.prof,'MarkerSize',ms);
    plot3([prof_lb(1) prof_ub(1) prof_ub(1) prof_lb(1) prof_lb(1)], [prof_lb(2) prof_lb(2) prof_ub(2) prof_ub(2),prof_lb(2)],c*ones(5,1),'-','Color',colors.prof,'LineWidth',1,'DisplayName','Profile bounds');
    
    if isempty(fixedI)
        ax=subplot(234);
        hold('on')
        title('I')
        ax.YDir='reverse';
        ax.XAxisLocation='bottom';
        ax.TickDir='out';
        ax.XMinorTick='on';
        ax.YMinorTick='on';
        ax.ZMinorTick='on';
        ax.XGrid='on';
        ax.YGrid='on';
        ax.ZGrid='on';
        ax.Box='on';
        ax.BoxStyle='full';
        ax.Projection='Orthographic';
        view(23,13);
        pbaspect(aspect);
        xlabel('x (px)');
        ylabel('y (px)');
        zlabel('I(photons)');
        I = reshape(prof_params(3,:),N,N);
        ax.Color=[0.1,0.1,0.1];
        colormap(ax,'parula')
        contour3(X,Y,I,30);
    end

    ax=subplot(235);
    hold('on')
    title('bg')
    ax.YDir='reverse';
    ax.XAxisLocation='bottom';
    ax.TickDir='out';
    ax.XMinorTick='on';
    ax.YMinorTick='on';
    ax.ZMinorTick='on';
    ax.XGrid='on';
    ax.YGrid='on';
    ax.ZGrid='on';
    ax.Box='on';
    ax.BoxStyle='full';
    ax.Projection='Orthographic';
    view(23,13);
    pbaspect(aspect);
    xlabel('x (px)');
    ylabel('y (px)');
    zlabel('bg(phtons/px)');
    ax.Color=[0.1,0.1,0.1];
    bg = reshape(prof_params(4,:),N,N);
    colormap(ax,'parula')
    contour3(X,Y,bg,30);


    if obj.NumParams>=5
        ax=subplot(236);
        hold('on')
        title('sigma ratio')
        ax.YDir='reverse';
        ax.XAxisLocation='bottom';
        ax.TickDir='out';
        ax.XMinorTick='on';
        ax.YMinorTick='on';
        ax.ZMinorTick='on';
        ax.XGrid='on';
        ax.YGrid='on';
        ax.ZGrid='on';
        ax.Box='on';
        ax.BoxStyle='full';
        ax.Projection='Orthographic';
        view(23,13);
        pbaspect(aspect);
        xlabel('x (px)');
        ylabel('y (px)');
        zlabel('bg(phtons/px)');
        ax.Color=[0.1,0.1,0.1];
        bg = reshape(prof_params(5,:),N,N);
        colormap(ax,'parula')
        contour3(X,Y,bg,30);
    end
    
%     [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,fixed_params,xs,method,prof_params);
% %     [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,fixed_params,xs,method,prof_params+randn(4,N).*repmat([.5;.5;10;1],1,N));
% %     [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,fixed_params,xs,method,repmat(theta',1,N)+randn(4,N).*repmat([.5;.5;1000;10],1,N));
%     
%     %Normalize so that theta_mle=0;
%     init_rllh=init_rllh-mle_rllh;
%     prof_rllh=prof_rllh-mle_rllh;
%     marg_rllh=marg_rllh-mle_rllh;
%     theta_rllh=theta_rllh-mle_rllh;
%     theta_true_slice_rllh=theta_true_slice_rllh-mle_rllh;
%     
%     dx = xs-theta_mle(param_idx);
%     obs_llh_model=-.5*dx.^2*obsI(param_idx,param_idx);
%     
%     fig = figure();
%     subplot(1,2,1)
%     ms=5;
%     plot(theta(param_idx),theta_rllh,'bs','MarkerFaceColor',[0 0 .8],'MarkerSize',4,'DisplayName',sprintf('true llh%.9g',theta_rllh));
%     hold('on')
%     plot(theta_mle(param_idx),0,'mo','MarkerFaceColor',[1,.5,1],'MarkerSize',ms,'DisplayName',sprintf('theta mle llh%.9g',mle_rllh));
%     plot(theta_mle(param_idx),0,'b^','MarkerFaceColor',[1,.5,.5],'MarkerSize',ms,'DisplayName',sprintf('theta init llh%.9g',init_rllh));
%     plot([obs_lb(param_idx),obs_ub(param_idx)],c*[1,1],'x-','Color',[0,.2,.5],'DisplayName','Observed Bounds')
%     plot([exp_lb(param_idx),exp_ub(param_idx)],[1,1],'o-','Color',[1,.5,1],'DisplayName','Expected Bounds')
%     plot(xs,marg_rllh,'-','Color',[1,0,0],'LineWidth',2,'DisplayName','Sliced Likelihoood')
%     plot(xs,theta_true_slice_rllh,'--','Color',[0,0,1],'LineWidth',1,'DisplayName','True theta Sliced Likelihoood')
%     plot(xs,obs_llh_model,'-.','Color',[.3 0 .6],'LineWidth',1,'DisplayName','mle Observed information model')
%     plot(xs,prof_rllh,'-','Color',[0,0,0],'DisplayName','Profile Likelihoood')
%     plot(xs,c*ones(N,1),':','Color',[0,1,0],'DisplayName','Chi-sq likelhood ratio test')
%     legend('location','best')
%     ylabel('relative log-likelihood')
%     xlabel(obj.ParamNames{param_idx})

end
