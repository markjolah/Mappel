/** @file mcmc_data.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2015-2019
 * @brief MCMC data storage types
 */

#ifndef MAPPEL_MCMC_DATA_H
#define MAPPEL_MCMC_DATA_H

#include <armadillo>

namespace mappel {

namespace mcmc {

/** Structures for reporting MCMC results
 */
///@{
/** Data and controlling parameters for an MCMC posterior sampling for a single data.
 */
struct MCMCData {
    /* Controlling parameters */
    IdxT Nsample=0;
    IdxT Nburnin=0;
    IdxT thin=0;
    double confidence=-INFINITY;

    VecT sample_mean;
    MatT sample_cov;
    VecT credible_lb;
    VecT credible_ub;
    MatT sample;
    VecT sample_rllh;
    void initialize_arrays(IdxT Nparams);
};

/** Debugging Data and controlling parameters for an MCMC posterior sampling for a single data.
 * No burnin or thinning is performed when debugging
 */
struct MCMCDebugData {
    /* Controlling parameters */
    IdxT Nsample=0;

    MatT sample;
    VecT sample_rllh;
    MatT candidate;
    VecT candidate_rllh;
    void initialize_arrays(IdxT Nparams);
};

/** Data and controlling parameters for an MCMC posterior sampling for stack of single data.
 */
struct MCMCDataStack {
    /* Controlling parameters */
    IdxT Nsample=0;
    IdxT Nburnin=0;
    IdxT thin=0;
    double confidence=-INFINITY;

    /* Restult variables */
    IdxT Ndata=0;
    MatT sample_mean;
    CubeT sample_cov;
    MatT credible_lb;
    MatT credible_ub;
    CubeT sample;
    MatT sample_rllh;
    void initialize_arrays(IdxT Nparams);
};
///@}

} /* namespace mappel::mcmc */    
} /* namespace mappel */

#endif /* MAPPEL_MCMC_H */
