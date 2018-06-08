/** @file MCMCAdaptor2D.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class declaration and inline and templated functions for MCMCAdaptor2D.
 */

#ifndef _MAPPEL_MCMCADAPTOR2D_H
#define _MAPPEL_MCMCADAPTOR2D_H

#include "Mappel/MCMCAdaptorBase.h"
#include "Mappel/PointEmitterModel.h"

namespace mappel {

class MCMCAdaptor2D : public virtual PointEmitterModel, public MCMCAdaptorBase
{
public:
    void sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale=1.0);
protected:
    MCMCAdaptor2D();
    explicit MCMCAdaptor2D(double sigma_scale);
    MCMCAdaptor2D(const MCMCAdaptor2D &o);
    MCMCAdaptor2D(MCMCAdaptor2D &&o);
    MCMCAdaptor2D& operator=(const MCMCAdaptor2D &o);
    MCMCAdaptor2D& operator=(MCMCAdaptor2D &&o);    
    StatsT get_stats() const;
private:
    void initialize();

    double eta_x=0; /**< The standard deviation for the normally distributed perturbation to theta_x in the random walk MCMC sampling */
    double eta_y=0; /**< The standard deviation for the normally distributed perturbation to theta_y in the random walk MCMC sampling */
    double eta_I=0; /**< The standard deviation for the normally distributed perturbation to theta_I in the random walk MCMC sampling */
    double eta_bg=0; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
};

} /* namespace mappel */

#endif /* _MAPPEL_MCMCADAPTOR2D_H */
