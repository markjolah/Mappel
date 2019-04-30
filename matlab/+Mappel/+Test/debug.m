M = Mappel.Gauss2DsMAP(8,1, 3);
theta = [3.2,6.6,100,10,1+eps];
Nims = 100;
confidence = 0.95;
estimate_parameter = 1;
llh_delta = -chi2inv(.95,1)/2;
im = M.simulateImage(theta);
ims = M.simulateImage(theta);

[theta_mle, mle_rllh, obsI, est_stats] = M.estimateMax(im,'TrustRegion');
% 
% [profile_lb, profile_ub, stats]  = M.errorBoundsProfileLikelihood(im, theta_mle, confidence, mle_rllh, obsI);
% [profile_lb2, profile_ub2, stats2]  = M.errorBoundsProfileLikelihood(im, theta_mle, confidence);
% [db_profile_lb, db_profile_ub, seq_lb, seq_ub, seq_lb_rllh, seq_ub_rllh, db_stats] ...
%     = M.errorBoundsProfileLikelihoodDebug(im, theta_mle, obsI, estimate_parameter, llh_delta);
