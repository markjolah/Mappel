
function fig = profileViewer(obj, theta, im, param_idx, method)
    if nargin<5
        method='NewtonDiagonal';
    end
    confidence=0.95;
    c=-chi2inv(confidence,1)/2
    theta_rllh = obj.modelRLLH(im,theta);
    theta_init = obj.estimate(im,'Heuristic');
    init_rllh = obj.modelRLLH(im,theta_init);
    [theta_mle,mle_rllh,obsI,mle_stats] = obj.estimate(im,method,theta_init);
    crlb = obj.CRLB(theta);
    [exp_lb,exp_ub]=obj.errorBoundsExpected(theta_mle,confidence);
    [obs_lb,obs_ub]=obj.errorBoundsObserved(im,theta_mle,confidence,obsI);
    [min_lb,max_ub]=obj.errorBoundsExpected(theta_mle,0.999);
    N=100;
%     xs=linspace(0.6*min_lb(param_idx),1.5*max_ub(param_idx),N);
    xs=linspace(0.1,7.9,N);
    marg_thetas=repmat(theta_mle,1,N);
    marg_thetas(param_idx,:)=xs;
    marg_rllh = obj.modelRLLH(im,marg_thetas);
    
    theta_true_slice=repmat(theta(:),1,N);
    theta_true_slice(param_idx,:)=xs;
    theta_true_slice_rllh = obj.modelRLLH(im,theta_true_slice);
    
    
    fixed_params=zeros(obj.NumParams,1,'uint64');
    fixed_params(param_idx)=1;
    [prof_rllh,prof_params,stats]=obj.estimateProfileLikelihood(im,fixed_params,xs,method,theta);
    
    %Normalize so that theta_mle=0;
    init_rllh=init_rllh-mle_rllh;
    prof_rllh=prof_rllh-mle_rllh;
    marg_rllh=marg_rllh-mle_rllh;
    theta_rllh=theta_rllh-mle_rllh;
    theta_true_slice_rllh=theta_true_slice_rllh-mle_rllh;
    
    dx = xs-theta_mle(param_idx);
    obs_llh_model=-.5*dx.^2*obsI(param_idx,param_idx);
    
    fig = figure();
    subplot(1,2,1)
    ms=5;
    plot(theta(param_idx),theta_rllh,'bs','MarkerFaceColor',[0 0 .8],'MarkerSize',4,'DisplayName',sprintf('true llh%.9g',theta_rllh));
    hold('on')
    plot(theta_mle(param_idx),0,'mo','MarkerFaceColor',[1,.5,1],'MarkerSize',ms,'DisplayName',sprintf('theta mle llh%.9g',mle_rllh));
    plot(theta_mle(param_idx),0,'b^','MarkerFaceColor',[1,.5,.5],'MarkerSize',ms,'DisplayName',sprintf('theta init llh%.9g',init_rllh));
    plot([obs_lb(param_idx),obs_ub(param_idx)],c*[1,1],'x-','Color',[0,.2,.5],'DisplayName','Observed Bounds')
    plot([exp_lb(param_idx),exp_ub(param_idx)],[1,1],'o-','Color',[1,.5,1],'DisplayName','Expected Bounds')
    plot(xs,marg_rllh,'-','Color',[1,0,0],'LineWidth',2,'DisplayName','Sliced Likelihoood')
    plot(xs,theta_true_slice_rllh,'--','Color',[0,0,1],'LineWidth',1,'DisplayName','True theta Sliced Likelihoood')
    plot(xs,obs_llh_model,'-.','Color',[.3 0 .6],'LineWidth',1,'DisplayName','mle Observed information model')
    plot(xs,prof_rllh,'-','Color',[0,0,0],'DisplayName','Profile Likelihoood')
    plot(xs,c*ones(N,1),':','Color',[0,1,0],'DisplayName','Chi-sq likelhood ratio test')
    legend('location','best')
    ylabel('relative log-likelihood')
    xlabel(obj.ParamNames{param_idx})

end
