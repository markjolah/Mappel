
size=8;
psfsigma=1.0;
theta=[4.0, 5.2, 1250, 10, 1.6];
epsilon=1e-3;
blinks=[1.0, 1.0, 0.9, 0, 0, 0.8, 1, 1];
blinks=min(1-epsilon, max(epsilon, blinks));
theta=[theta blinks];
model=Blink2DsMAP(size,psfsigma);

model_im=model.modelImage(theta);
dip_model_im=model.makeDipImage(model_im);
im=model.simulateImage(theta);
dip_sim_im=model.makeDipImage(im);
max_px=max(max(max(im)),max(max(model_im)));

f1=figure(1);
dipshow(f1,dip_model_im, [0,max_px]);
dipmapping(f1,'colormap',hot(256));
diptruesize(f1,8000);

f2=figure(2);
dipshow(f2, dip_sim_im,[0,max_px]);
dipmapping(f2,'colormap',hot(256));
diptruesize(f2,8000);
hold on
%[theta_est, crlb, llh, stats]=model.estimateMAP(im,'NewtonRaphson')
% [theta_est, crlb, llh, stats, seq, seq_rllh]=model.estimateMAPDebug(im,'NewtonRaphson')
[pmean, pcov, sample, sample_rllh, pseq, pseq_rllh]=model.estimatePosteriorDebug(im,10000);
plot( pseq(1,:), pseq(2,:), 'g-o','markersize',4,'markerfacecolor','g','markeredgecolor',[0.0,0.5,0.05] ,'linewidth',4);
plot( sample(1,:), sample(2,:), 'y-o','markersize',4,'markerfacecolor','y','markeredgecolor',[1.0,0.5,0.00] ,'linewidth',4);
[theta_est, crlb, llh, stats, seq, seq_rllh]=model.estimateMAPDebug(im,'NewtonRaphson')
plot( seq(1,1:end-1), seq(2,1:end-1), 'b-o','markersize',4,'markerfacecolor','b' ,'markeredgecolor',[0.0,0.05,0.5],'linewidth',4);

plot( theta(1), theta(2), 'p','markerfacecolor',[1.0,0.1,1.0],'markeredgecolor','k','linewidth',3,'MarkerSize',35);
plot( pmean(1), pmean(2), 'd','markerfacecolor',[0.0,1.0,0.5],'linewidth', 3, 'markeredgecolor','k','MarkerSize',20)
plot( theta_est(1), theta_est(2), 's','markerfacecolor',[0.2,0.1,1.0],'linewidth', 3,'markeredgecolor','k' ,'MarkerSize',20)

htheta_est=model.estimateMAP(im,'Huristic');
plot( htheta_est(1), htheta_est(2), 'o','markerfacecolor',[0.0,0.2,0.2],'linewidth', 3,'markeredgecolor','k' ,'MarkerSize',5)


hold off

f3=figure(3);
plot(1:length(pseq_rllh), pseq_rllh, '-k');


