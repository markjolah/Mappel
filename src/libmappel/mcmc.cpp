#include "mcmc.h"

namespace mappel {

MatT thin_sample(MatT &sample, int burn_in, int keep_every)
{
    int nkeep=(sample.n_cols-burn_in)/keep_every;
    MatT subsample(sample.n_rows,nkeep);
    for(unsigned n=burn_in; n<sample.n_cols; n+=keep_every){
        subsample.col((n-burn_in)/keep_every)=sample.col(n);
    }
    return subsample;
}

} /* namespace mappel */
