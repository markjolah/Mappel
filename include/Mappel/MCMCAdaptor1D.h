/** @file MCMCAdaptor1D.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class declaration and inline and templated functions for MCMCAdaptor1D.
 */

#ifndef _MAPPEL_MCMCADAPTOR1D_H
#define _MAPPEL_MCMCADAPTOR1D_H

#include "Mappel/MCMCAdaptorBase.h"
#include "Mappel/PointEmitterModel.h"

namespace mappel {

class MCMCAdaptor1D : public virtual PointEmitterModel, public MCMCAdaptorBase
{
public:
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale=1.0);
protected:
    MCMCAdaptor1D();
    explicit MCMCAdaptor1D(double sigma_scale);
    MCMCAdaptor1D(const MCMCAdaptor1D &o);
    MCMCAdaptor1D(MCMCAdaptor1D &&o);
    MCMCAdaptor1D& operator=(const MCMCAdaptor1D &o);
    MCMCAdaptor1D& operator=(MCMCAdaptor1D &&o);    
    StatsT get_stats() const;
private:
    void initialize();

    double eta_x=0; /**< The standard deviation for the normally distributed perturbation to theta_x in the random walk MCMC sampling */
    double eta_I=0; /**< The standard deviation for the normally distributed perturbation to theta_I in the random walk MCMC sampling */
    double eta_bg=0; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
};

  
} /* namespace mappel */

#endif /* _MAPPEL_MCMCADAPTOR1D_H */
