/** @file MCMCAdaptorBase.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class declaration and inline and templated functions for MCMCAdaptorBase.
 */

#ifndef _MAPPEL_MCMCADAPTORBASE_H
#define _MAPPEL_MCMCADAPTORBASE_H

#include "Mappel/util.h"
namespace mappel {

class MCMCAdaptorBase
{
public:
    static const double global_default_mcmc_sigma_scale;// = 0.05
    static const double global_max_mcmc_sigma_scale;// = 0.5
    
    void set_mcmc_sigma_scale(double scale);
    double get_mcmc_sigma_scale() const;
    IdxT get_mcmc_num_phases() const;

protected:
    MCMCAdaptorBase(IdxT num_phases);
    MCMCAdaptorBase(IdxT num_phases, double sigma_scale);
    void set_mcmc_num_phases(IdxT num_phases);

    StatsT get_stats() const;
    IdxT num_phases; /**< The number of different sampling phases for candidate selection MCMC.  Each phase changes a different subset of variables.*/
    double sigma_scale; /**< A scaling factor for step sizes as a fraction of the size of the domain dimension we are walking in. (0.05 default)*/
};

  
} /* namespace mappel */

#endif /* _MAPPEL_MCMCADAPTORBASE_H */
