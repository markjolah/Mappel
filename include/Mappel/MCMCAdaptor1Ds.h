/** @file MCMCAdaptor1Ds.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018-2019
 * @brief The class declaration and inline and templated functions for MCMCAdaptor1Ds.
 */

#ifndef MAPPEL_MCMCADAPTOR1DS_H
#define MAPPEL_MCMCADAPTOR1DS_H

#include "Mappel/MCMCAdaptor1D.h"

namespace mappel {

class MCMCAdaptor1Ds : public MCMCAdaptor1D
{
public:
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale=1.0);
protected:
    MCMCAdaptor1Ds();
    explicit MCMCAdaptor1Ds(double sigma_scale);
    MCMCAdaptor1Ds(const MCMCAdaptor1Ds &o);
    MCMCAdaptor1Ds(MCMCAdaptor1Ds &&o);
    MCMCAdaptor1Ds& operator=(const MCMCAdaptor1Ds &o);
    MCMCAdaptor1Ds& operator=(MCMCAdaptor1Ds &&o);    
    StatsT get_stats() const;

    double eta_sigma=-1; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
};

} /* namespace mappel */

#endif /* MAPPEL_MCMCADAPTOR1DS_H */
