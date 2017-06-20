/** @file PointEmitterHSModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for PointEmitterHSModel
 */

#include "PointEmitterHSModel.h"
#include "util.h"

const std::vector<std::string> PointEmitterHSModel::estimator_names(
    { "HeuristicMLE", "NewtonMLE", "NewtonRaphsonMLE", "QuasiNewtonMLE", "SimulatedAnnealingMLE"});

PointEmitterHSModel::PointEmitterHSModel(int num_params, const IVecT &size, const VecT &sigma)
    : PointEmitterModel(num_params), ndim(3),
      size(size), psf_sigma(2),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      L_dist(BetaRNG(beta_L,beta_L)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_L_const(log_prior_beta_const(beta_L)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg))
{
    log_prior_const=2*log_prior_pos_const+log_prior_L_const+log_prior_I_const+log_prior_bg_const;
    assert(size.n_elem==3);
    assert(sigma.n_elem==3);

    psf_sigma(0)=sigma(0);
    psf_sigma(1)=sigma(1);
    mean_sigmaL=sigma(2);

    candidate_eta_x=size(0)*candidate_sample_dist_ratio;
    candidate_eta_y=size(1)*candidate_sample_dist_ratio;
    candidate_eta_L=size(2)*candidate_sample_dist_ratio;
    candidate_eta_I=mean_I*candidate_sample_dist_ratio;
    candidate_eta_bg=mean_bg*candidate_sample_dist_ratio;
}

PointEmitterHSModel::StatsT
PointEmitterHSModel::get_stats() const
{
    StatsT stats;
    stats["dimensions"]=ndim;
    stats["sizeX"]=size(0);
    stats["sizeY"]=size(1);
    stats["sizeL"]=size(2);
    stats["psfSigmaX"]=psf_sigma(0);
    stats["psfSigmaY"]=psf_sigma(1);
    stats["meanSigmaL"]=mean_sigmaL;
    stats["numParams"]=num_params;
    stats["Xmin"]=0;
    stats["Xmax"]=size(0);
    stats["Ymin"]=0;
    stats["Ymax"]=size(1);
    stats["Lmin"]=0;
    stats["Lmax"]=size(2);
    stats["candidate.etaX"]=candidate_eta_x;
    stats["candidate.etaY"]=candidate_eta_y;
    stats["candidate.etaL"]=candidate_eta_L;
    stats["candidate.etaI"]=candidate_eta_I;
    stats["candidate.etabg"]=candidate_eta_bg;
    return stats;
}
