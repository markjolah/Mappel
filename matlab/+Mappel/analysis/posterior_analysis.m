
imsize=[8,8];
psfsigma=1.0;
x=3.8;
y=5.4;
I=2000;
bg=5.0;
sigma=1.3;
epsilon=1e-3;
theta=[x,y,I, bg, sigma]; 
blinks=[1, 1.0, 0.1, 0.2, 1.0, 0.7, 1, 1.];
% blinks=[1.0, 0.7, 0.0, 0.7];
blinks=min(1-epsilon, max(epsilon, blinks));
gentheta=[theta blinks]';
genmodel=Blink2DsMAP(imsize,psfsigma);
model=Blink2DsMAP(imsize,psfsigma);
theta=gentheta(1:model.nParams);

% theta=[x,y,I, bg,sigma]'; 
% model=Gauss2DsMAP(size,psfsigma);
% theta=model.samplePrior(1);

N=50000;
im=genmodel.simulateImage(gentheta);
[pmean, pcov, seq, seq_rllh, can, can_rllh]=model.estimatePosteriorDebug(im,N);
[est_theta, est_crlb, est_llh, stats, map_seq, map_seq_rllh]=model.estimateDebug(im,'NewtonRaphson');

hest_theta=model.estimate(im,'Heuristic');
map_error=theta-est_theta;
map_rmse=sqrt(sum(map_error(1:2,:).*map_error(1:2,:)));

f=figure(1);
clf
model.viewDipImage(im,f);
hold on;
plot( can(1,:)-0.5, can(2,:)-0.5, 'b-o','markersize',2,'markerfacecolor','b','markeredgecolor',[0.0,0.05,0.5] ,'linewidth',1);
plot( seq(1,:)-0.5, seq(2,:)-0.5, 'g-o','markersize',3,'markerfacecolor','g','markeredgecolor',[0.0,0.5,0.05] ,'linewidth',2);
plot( pmean(1)-0.5, pmean(2)-0.5, 'd','markerfacecolor',[0.0,0.7,0.0],'linewidth', 1, 'markeredgecolor','k','MarkerSize',15)
plot( theta(1)-0.5, theta(2)-0.5, 'p','markerfacecolor','none','markeredgecolor',[1.0,0.1,1.0],'linewidth',2,'MarkerSize',25);
plot( map_seq(1,:)-0.5, map_seq(2,:)-0.5, '-o','color',[.0,0.5,1.0], 'markersize',3,'markerfacecolor',[.0,0.5,1.0],'markeredgecolor',[.0,0.5,1.0] ,'linewidth',3);
plot( est_theta(1)-0.5, est_theta(2)-0.5, 'v','markerfacecolor',[.0,0.5,1.0],'linewidth', 1, 'markeredgecolor','k','MarkerSize',15)
plot( hest_theta(1)-0.5, hest_theta(2)-0.5, 'o','markerfacecolor',[0.8,0.9,0],'linewidth', 1, 'markeredgecolor','k','MarkerSize',10)
hold off;

f=figure(2);
clf(2);
thin=1;
xs=1:thin:N;
seq=seq(:,1:thin:N);
cummeans=cumsum(seq')'./repmat(1:N/thin,model.nParams,1);
hold on;
plot(xs, can(1,:), 'or','MarkerSize',2);
plot(xs,cummeans(1,:),'-','linewidth',2,'Color',[0 0 1]);
plot([1 N], [x x], 'o-.','linewidth',2,'Color',[0.5,0,0]);
plot(xs, seq(1,:), '-','linewidth',1,'Color',[0 0 0]);

% plot(xs,cummeans(2,:),'b--','linewidth',2);
% plot([1 N], [y y], 'o-.','linewidth',2,'Color',[0,0,0.5]);
% plot(xs, seq(2,:), '-','linewidth',2,'Color',[0,0,1.0]);
% plot(xs, can(2,:), 'ob','MarkerSize',2);
title('Position Approximation');
legend('(x) Candidate Samples','(x) Posterior Mean', '(x) True Value', '(x) Accepted Samples')
%        '(y) Posterior Mean', '(y) True Value', '(y) Accepted Samples', '(y) Candidate Samples');
% axes()
% plot([1 N], [I I], '--g');
% ylim(ax, [0 8]);
% set(ax,'YTick',0:8)
xlabel('Samples');
ylabel('estimated posterior mean');
hold off;
% hold(axs(2));

f=figure(3);
clf
acceptance= sum(abs(seq(:,2:end)-seq(:,1:end-1)))>0;
acceptance_rate=cumsum(double(acceptance))./ (1:N-1);
hold on;
plot(xs, can_rllh, 'ob','MarkerSize',2);
plotyy(xs,seq_rllh,xs(1:end-1),acceptance_rate);
hold off;
xlabel('Samples');
ylabel('RLLH');
% plot(xs,cummeans(3,:),'I')

f=figure(4);
clf
plot([1 N], [sigma sigma], '--k','Color',[0.0,0.5,0.5],'linewidth',2);
hold on;
plot(xs, seq(5,:), ':k','Color',[0.5,0,0.5],'linewidth',2);
plot(xs, can(5,:), 'ok','MarkerSize',2);
plot(xs,cummeans(5,:),'-','Color',[0.5,0,0.5],'linewidth',2);
xlabel('Samples');
ylabel('sigma');
hold off;


f=figure(5);
clf
trim=100;
trimmed_xs=1:N-trim;
trimmed_seq=seq(:,trim+1:end);
trimmed_cummeans=cumsum(trimmed_seq')' ./repmat(trimmed_xs,model.nParams,1);
trimmed_errors=trimmed_cummeans-repmat(theta,1,N-trim);
trimmed_rmse=sqrt(sum(trimmed_errors.*trimmed_errors));
trimmed_pos_rmse=sqrt(sum(trimmed_errors(1:2,:).*trimmed_errors(1:2,:)));

errors=cummeans-repmat(theta,1,N);
% rmse=sqrt(mean(errors.*errors,1));
pos_rmse=sqrt(errors(1,:).*errors(1,:));
ax=axes();
set(ax,'XScale','linear')
set(ax,'YScale','linear')
hold on;

% plot(xs,rmse,'k:');
plot(xs,pos_rmse,'r-');
plot(N,sqrt(pcov(1,1)),'b*');
% plot(trimmed_xs,trimmed_rmse,'k-');
% plot(trimmed_xs,trimmed_pos_rmse,'r-');


plot([0,N],[map_rmse,map_rmse],'--k','Color',[0.8,0.3,0.0],'linewidth',2);
lim=ylim;
ylim([0,lim(2)]);
hold off;
xlabel('Samples')
ylabel('RMSE Posterior Mean Estimate')

f=figure(6);
model.viewDipImage(model.modelImage(theta),f);
set(f,'Name','Model Image');

