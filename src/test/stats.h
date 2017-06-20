/**
 * @file stats.h
 * 
 * @author Mark J. Olah (email mjo@cs.unm.edu )
 * @date 12-12-2013
 * 
 * @brief Probability and statistical calculations
 */
#ifndef _STATS_H
#define _STATS_H
#include <armadillo>

/** @brief 
 * 
 * 
 */
double test_mean_nD_one_sided(int sample_size, const arma::vec &sample_mean,
                             const arma::mat &sample_covariance, const arma::vec &mu);

template <class Vec>
inline double test_mean_nD_one_sided(arma::running_stat_vec<Vec> sample, const arma::vec &mu){
    return test_mean_nD_one_sided(sample.count(),sample.mean(),sample.cov(),mu);
}
#endif /* _STATS_H */
