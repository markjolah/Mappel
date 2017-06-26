function intializationTest(theta, method)
    if nargin==1
        method = 'newton';
    end
    sz=[8 8];
    psf=[1 1];
    g2d=Gauss2DMAP(sz, psf);
    im=g2d.simulateImage(theta);
    N=1e3;
    im = repmat(im,1,1,N);
%     theta_init = [rand(2,N)*sz(1); gamrnd(2,1e3/2,1,N); gamrnd(2,5e0/2,1,N)];
    theta_init = [rand(2,N)*sz(1); repmat(theta([3,4]),N,1)' ];
    [theta_e, crlb, llh] = g2d.estimate(im,method,theta_init);
    fails = find( abs(theta_e(1,:)-repmat(theta(1),1,N))>0.5);
    success = find( abs(theta_e(1,:)-repmat(theta(1),1,N))<=0.5);
    [~,si] = sort(theta_init(1,:));

    figure();
    subplot(2,2,1);
    quiver(theta_init(1,fails),theta_init(2,fails),theta_e(1,fails),theta_e(2,fails),'r');
    hold on;
%     quiver(theta_init(1,success),theta_init(2,success),theta_e(1,success),theta_e(2,success),'g');
    xlim([0 sz(1)]);
    ylim([0 sz(2)]);
    subplot(2,2,2);
    scatter(theta_e(1,:), theta_e(2,:));
    hold on;
    scatter(theta(1),theta(2),'gx');
    xlim([0 sz(1)]);
    ylim([0 sz(2)]);
    subplot(2,2,3);
    imagesc(im(:,:,1));
    set(gca(),'YDir','normal');
    subplot(2,2,4);
    scatter(theta_init(1,fails),theta_init(2,fails), 'rx');
    hold on;
    scatter(theta_init(1,success),theta_init(2,success), 'go');
    xlim([0 sz(1)]);
    ylim([0 sz(2)]);
%     plot(theta_init(1,si), llh(si));

%     hist3(theta_e([1,2],:)');
%     subplot(2,1,2);
%     hist3(theta_e([3,4],:)');

end
