#include <boost/math/distributions/fisher_f.hpp>
#include "stats.h"


double test_mean_nD_one_sided(int sample_size, const arma::vec &sample_mean,
                             const arma::mat &sample_covariance, const arma::vec &mu)
{
    using boost::math::fisher_f_distribution;
    if (arma::det(sample_covariance)==0.) return 0.; //This takes care of the case when all samples are equal
    double n=sample_size; //defined for clarity
    double p=mu.n_elem;   //defined for clarity
    arma::vec sample_mean_error=sample_mean-mu;
    double f_stat= (n-p)/(p*(n-1)) * n *
        (sample_mean_error.t()*sample_covariance.i()*sample_mean_error).eval()[0];
    fisher_f_distribution<double> F(p,n-p);
    double p_val=cdf(complement(F, f_stat));
    return p_val;
}
