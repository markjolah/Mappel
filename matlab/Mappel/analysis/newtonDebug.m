function im=newtonDebug(theta, theta_init)
    sz=[8 8];
    psf=[1 1];
    g2d=Gauss2DsMAP(sz, psf);
    im=g2d.simulateImage(theta);
    Heuristic=struct();
    [Heuristic.theta, Heuristic.crlb, Heuristic.llh, Heuristic.stats] = g2d.estimate(im,'Heuristic',theta_init);
    Newton=struct();
    [Newton.theta, Newton.crlb, Newton.llh, Newton.stats, Newton.sample, Newton.sample_llh] = g2d.estimateDebug(im,'Newton',theta_init);
%     NewtonRaphson=struct();
%     [NewtonRaphson.theta, NewtonRaphson.crlb, NewtonRaphson.llh, NewtonRaphson.stats,...
%          NewtonRaphson.sample, NewtonRaphson.sample_llh] = g2d.estimateDebug(im,'NewtonRaphson',theta_init);

    

    Npoints=100;
    xs=linspace(0,sz(1),Npoints);
    ys=linspace(0,sz(2),Npoints);
    [X, Y] = meshgrid(xs,ys);
    thetas = [X(:), Y(:), repmat(theta(3:end),Npoints^2,1)]';
    LLHs = g2d.LLH(im, thetas);
    
    H=g2d.modelHessian(im, Heuristic.theta);
    G=g2d.modelGrad(im, Heuristic.theta);
    C=g2d.modelPositiveHessian(im, Heuristic.theta);
    C = C-diag(diag(C))+diag(sqrt(diag(C)));
    C2=cholmod(-H);
    step.newton = -H\G;
    step.nr = -G./diag(H);
    step.chol = C\G;
    chol2Step = inv(C2'*C2)*G;
    scale = 1./sqrt(abs(diag(H)));
    S = diag(scale);
    HS = S'*H*S;
    HS = triu(HS)+triu(HS)'-diag(diag(HS)); %force to be symmetric
    GS = S*G;    
    CS = cholmod(HS);
    
    step.newtonScaled = - HS\GS;
    step.newtonScaledChol = (CS'*CS)\GS;
    step.optimal = theta - theta_init;
    pt=Heuristic.theta;
    npt=pt+newtonStep;
    nllh = g2d.LLH(im,npt);
    nrpt=pt+nrStep;
    nrllh = g2d.LLH(im,nrpt);
    Newton.stats
    
    [m, fval, exitflag, output, grad, hessian] = fminunc(@(x) -g2d.LLH(im,x), theta_init);
    m
    -fval
    -hessian
    H=g2d.modelHessian(im, m);
    H+triu(H,1)'
    
    figure(1);
    clf();
    surface(X,Y, reshape(LLHs,Npoints,Npoints),'EdgeColor','None','FaceAlpha',0.8);
    hold('on');
    plot3(Heuristic.theta(1), Heuristic.theta(2), Heuristic.llh,'bs','MarkerFaceColor','b','MarkerSize',5);
    plot3(Newton.theta(1), Newton.theta(2), Newton.llh,'k^','MarkerFaceColor','k','MarkerSize',6);
    plot3(NewtonRaphson.theta(1), NewtonRaphson.theta(2), NewtonRaphson.llh,'ro','MarkerFaceColor','r','MarkerSize',4);
    plot3(m(1), m(2), -fval,'g>','MarkerFaceColor','g','MarkerSize',4);
    plot3([pt(1), npt(1)], [pt(2), npt(2)], [Heuristic.llh, nllh],'k-');
    plot3([pt(1), nrpt(1)], [pt(2), nrpt(2)], [Heuristic.llh, nrllh],'r-');
    hold('off');
    f=figure(2);
    dipshow(f,im);
end

