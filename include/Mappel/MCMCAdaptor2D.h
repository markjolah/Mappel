/** @file MCMCAdaptor2D.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018-2019
 * @brief The class declaration and inline and templated functions for MCMCAdaptor2D.
 */

#ifndef MAPPEL_MCMCADAPTOR2D_H
#define MAPPEL_MCMCADAPTOR2D_H

#include "Mappel/MCMCAdaptor1D.h"

namespace mappel {

class MCMCAdaptor2D : public MCMCAdaptor1D
{
public:
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale=1.0) const;
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_parameters_mask, double step_scale=1.0) const;
protected:
    MCMCAdaptor2D();
    explicit MCMCAdaptor2D(double sigma_scale);
    MCMCAdaptor2D(const MCMCAdaptor2D &o);
    MCMCAdaptor2D(MCMCAdaptor2D &&o);
    MCMCAdaptor2D& operator=(const MCMCAdaptor2D &o);
    MCMCAdaptor2D& operator=(MCMCAdaptor2D &&o);    
    StatsT get_stats() const;

    double eta_y=0; /**< The standard deviation for the normally distributed perturbation to theta_y in the random walk MCMC sampling */
};

} /* namespace mappel */

#endif /* MAPPEL_MCMCADAPTOR2D_H */
