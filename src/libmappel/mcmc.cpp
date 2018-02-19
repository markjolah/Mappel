// #include "mcmc.h"
#include "util.h"

namespace mappel {
namespace mcmc {

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
