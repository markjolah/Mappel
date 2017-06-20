/** @file BlinkModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 05-20-2014
 * @brief The class definition and template Specializations for BlinkModel
 */
#include "BlinkModel.h"
#include "stencil.h"

BlinkModel::BlinkModel(double candidate_sample_dist_ratio)
    : D_dist(BetaRNG(beta_D1,beta_D0)),
      log_prior_D_const(log_prior_beta2_const(beta_D0,beta_D1))
{
    candidate_eta_D=1.0*candidate_sample_dist_ratio;
}
