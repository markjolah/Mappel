
function testNewton()
    m=Gauss2DsMLE([12,12],1);

    theta = [6.6 5.2 1200 10 2.5];
    estimators = {'Newton', 'NewtonRaphson','SimulatedAnnealing'};
    Nestimators= numel(estimators);
    theta_init = [6 6 1000 10 3];

    im = m.simulateImage(theta);

    H=m.modelHessian(im,theta_init);
    g=m.modelGrad(im,theta_init);
    [mL,mD, is_positive] = Gauss2DMLE.modifiedCholesky(-H);
    H0 = mL*mD*mL';

    %chol(-H);
    stepN = H\-g;
    dirN = stepN' * g;
    stepC = H0\g;
    dirC = stepC' * g;
    stepC2 = Gauss2DMLE.choleskySolve(-H,g);
    dirC2 = stepC2' * g;
    f=figure('Position',[10,10,600,900],'Units','pixels');


    LLHS=cell(Nestimators,1);
    for n=1:Nestimators
        [etheta, crlb, llh, stats, seq, seq_llh] = m.estimateDebug(im,estimators{n},theta_init);
        ns = 1:length(seq);
        subplot(5,Nestimators,n);
        plot(ns, seq(1,:),'-r','DisplayName','x');
        hold('on');
        plot(ns, seq(2,:),'-b','DisplayName','y');
        plot([ns(1),ns(end)], [theta(1), theta(1)],':r','DisplayName','true-x');
        plot([ns(1),ns(end)], [theta(2), theta(2)],':b','DisplayName','true-y');
        title(sprintf('Estimator: %s', estimators{n}));
%         h = legend('location','best');
%         set(h,'interpreter','latex');
        xlabel('Sequence');
        if n==1
            ylabel('Position (pixels)');
        end

        subplot(5,Nestimators,n+Nestimators);
        plot(ns, seq(3,:),'-k','DisplayName','I');
        hold('on');
        plot([ns(1),ns(end)], [theta(3), theta(3)],':k','DisplayName','true-I');
%         h = legend('location','best');
%         set(h,'interpreter','latex');
        xlabel('Sequence');
        if n==1
            ylabel('Intensity (photons)');
        end

        subplot(5,Nestimators,n+Nestimators*2);
        plot(ns, seq(4,:),'DisplayName','bg');
        hold('on');
        plot([ns(1),ns(end)], [theta(4), theta(4)],':k','DisplayName','true-bg');
%         h = legend('location','best');
%         set(h,'interpreter','latex');
        xlabel('Sequence');
        if n==1
            ylabel('Background (photons/px)');
        end

        subplot(5,Nestimators,n+Nestimators*3);
        plot(ns, seq(5,:),'DisplayName','sigma');
        hold('on');
        plot([ns(1),ns(end)], [theta(5), theta(5)],':k','DisplayName','true-sigma');
%         h = legend('location','best');
%         set(h,'interpreter','latex');
%         xlabel('Sequence');
        if n==1
           ylabel('Gaussian Sigma (px)');
        end

        LLH{n}=seq_llh;
    end
    subplot(5,1,5);
    hold('on');
    for n=1:Nestimators
        plot(1:numel(LLH{n}), LLH{n},'DisplayName',sprintf('LLH_%s($\\theta$)',estimators{n}));
    end
    h = legend('location','best');
    set(h,'interpreter','latex');
    xlabel('Sequence');
    ylabel('LLH');
end
