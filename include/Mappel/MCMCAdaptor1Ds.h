/** @file MCMCAdaptor1Ds.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class declaration and inline and templated functions for MCMCAdaptor1Ds.
 */

#ifndef _MAPPEL_MCMCADAPTOR1DS_H
#define _MAPPEL_MCMCADAPTOR1DS_H

#include "Mappel/MCMCAdaptorBase.h"
#include "Mappel/PointEmitterModel.h"

namespace mappel {

class MCMCAdaptor1Ds : public virtual PointEmitterModel, public MCMCAdaptorBase
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
private:
    void initialize();

    double eta_x=0; /**< The standard deviation for the normally distributed perturbation to theta_x in the random walk MCMC sampling */
    double eta_I=0; /**< The standard deviation for the normally distributed perturbation to theta_I in the random walk MCMC sampling */
    double eta_bg=0; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
    double eta_sigma=0; /**< The standard deviation for the normally distributed perturbation to theta_bg in the random walk MCMC sampling */
};

} /* namespace mappel */

#endif /* _MAPPEL_MCMCADAPTOR1DS_H */
