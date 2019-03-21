/** @file mcmc.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief MCMC helper functions
 */
#include "Mappel/util.h"
#include "Mappel/mcmc_data.h"

namespace mappel {
namespace mcmc {

void MCMCData::initialize_arrays(IdxT Nparams)
{
    sample_mean.set_size(Nparams);
    sample_cov.set_size(Nparams,Nparams);
    credible_lb.set_size(Nparams);
    credible_ub.set_size(Nparams);
    sample.set_size(Nparams,Nsample);
    sample_rllh.set_size(Nsample);
}

void MCMCDebugData::initialize_arrays(IdxT Nparams)
{
    sample.set_size(Nparams,Nsample);
    sample_rllh.set_size(Nsample);
    candidate.set_size(Nparams,Nsample);
    candidate_rllh.set_size(Nsample);
}

void MCMCDataStack::initialize_arrays(IdxT Nparams)
{
    sample_mean.set_size(Nparams,Ndata);
    sample_cov.set_size(Nparams,Nparams,Ndata);
    credible_lb.set_size(Nparams,Ndata);
    credible_ub.set_size(Nparams,Ndata);
    sample.set_size(Nparams,Nsample,Ndata);
    sample_rllh.set_size(Nsample,Ndata);
}

IdxT num_oversample(IdxT Nsample, IdxT Nburnin, IdxT thin)
{
    if(thin==0) thin = 1;
    return 1+(Nsample-1)*thin + Nburnin;
}

MatT thin_sample(MatT &sample, IdxT Nburnin, IdxT thin)
{
    if(thin==0) thin = 1;
    IdxT Noversample = sample.n_cols;
    IdxT Nkeep = 1 + (Noversample-Nburnin-1)/thin;
    MatT subsample(sample.n_rows,Nkeep);
    IdxT k=0;
    for(IdxT n=Nburnin; n<Noversample; n+=thin) subsample.col(k++) = sample.col(n);
    return subsample;
}

void thin_sample(const MatT &sample, const VecT &sample_rllh, IdxT Nburnin, IdxT thin, MatT &subsample, VecT &subsample_rllh)
{
    if(thin==0) thin = 1;
    IdxT Noversample = sample.n_cols;
    IdxT Nkeep = 1 + (Noversample-Nburnin-1)/thin;
    subsample.set_size(sample.n_rows,Nkeep);
    subsample_rllh.set_size(Nkeep);
    IdxT k=0;
    for(IdxT n=Nburnin; n<Noversample; n+=thin) {
        subsample.col(k) = sample.col(n);
        subsample_rllh(k) = sample_rllh(n);
        k++;
    }
}

} /* namespace mappel::mcmc */
} /* namespace mappel */
