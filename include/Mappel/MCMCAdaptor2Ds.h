/** @file MCMCAdaptor2Ds.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018-2019
 * @brief The class declaration and inline and templated functions for MCMCAdaptor2Ds.
 */

#ifndef MAPPEL_MCMCADAPTOR2DS_H
#define MAPPEL_MCMCADAPTOR2DS_H

#include "Mappel/MCMCAdaptor2D.h"

namespace mappel {

class MCMCAdaptor2Ds : public MCMCAdaptor2D
{
public:
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale=1.0) const;
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_parameters_mask, double step_scale=1.0) const;
protected:
    MCMCAdaptor2Ds();
    explicit MCMCAdaptor2Ds(double sigma_scale);
    MCMCAdaptor2Ds(const MCMCAdaptor2Ds &o);
    MCMCAdaptor2Ds(MCMCAdaptor2Ds &&o);
    MCMCAdaptor2Ds& operator=(const MCMCAdaptor2Ds &o);
    MCMCAdaptor2Ds& operator=(MCMCAdaptor2Ds &&o);    
    StatsT get_stats() const;
    double eta_sigma=0; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
};

} /* namespace mappel */

#endif /* MAPPEL_MCMCADAPTOR2DS_H */
