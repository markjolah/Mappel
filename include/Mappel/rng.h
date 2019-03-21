/**
 * @file rng.h
 * 
 * @author Mark J. Olah (email mjo\@cs.unm DOT edu )
 * @date 2013-2019
 * 
 * @brief Random number generation usign sfmt
 */
#ifndef MAPPEL_RNG_H
#define MAPPEL_RNG_H

#include <random>

#include "Mappel/util.h"

#include <trng/lcg64_shift.hpp>
#include <ParallelRngManager/ParallelRngManager.h>

namespace mappel {

using ParallelRngGeneratorT = trng::lcg64_shift;
using ParallelRngManagerT = parallel_rng::ParallelRngManager<ParallelRngGeneratorT>;
using RngSeedT = parallel_rng::SeedT;
using UniformDistT = std::uniform_real_distribution<double>;

//Globals

//Single global rng manager
extern ParallelRngManagerT rng_manager;

/** @brief Generates a single Poisson distributed int from distribution with mean mu.
 * @param mu - mean of Poisson distribution
 * @param sfmt - A pointer to the SFMT rng state.
 * 
 * Knuth method circa 1969.  Transformed to work in log space.  This is linear in mu.  Works ok for small
 * counts.
 */
// template<class RngT>
// IdxT generate_poisson(RngT &rng, double mu);


template<class RngT>
IdxT generate_poisson_small(RngT &rng, double mu)
{
    UniformDistT uniform;
    IdxT k=0;
    double L=exp(-mu);
    if (L>=1.0) return 0;
    for(double p=1.0; p>L; k++) p *= uniform(rng);
    return k-1;
}

// "Rejection method PA" from "The Computer Generation of Poisson Random Variables" by A. C. Atkinson
// Journal of the Royal Statistical Society Series C (Applied Statistics) Vol. 28, No. 1. (1979)
// The article is on pages 29-35. The algorithm given here is on page 32.
template<class RngT>
IdxT generate_poisson_large(RngT &rng, double mu)
{
    UniformDistT uniform;
    double c = 0.767 - 3.36/mu;
    double beta = arma::datum::pi/sqrt(3.0*mu);
    double alpha = beta*mu;
    double k = log(c) - mu - log(beta);
    
    while(true) {
        double u = 1-uniform(rng);//support: (0,1]
        double x = (alpha - log((1.0 - u)/u))/beta;
        double n = floor(x + 0.5);
        if(n<0) continue;
        double v = 1-uniform(rng);//support: (0,1]
        double y = alpha - beta*x;
        double temp = 1.0 + exp(y);
        double lhs = y + log(v/(temp*temp));
        double rhs = k + n*log(mu) - lgamma(n+1);
        if (lhs <= rhs)
            return n;
    }
}

template<class RngT>
double generate_poisson(RngT &rng, double mu)
{
    const uint64_t max_mu = std::numeric_limits<uint32_t>::max()/4;
    if (mu<0. || !std::isfinite(mu)) {
        std::ostringstream msg;
        msg<<"Generate Poisson got invalid rate mu:"<<mu;
        throw ParameterValueError(msg.str());
    } else if (mu == 0.) {
        return 0.;
    } else if (mu < 30.) {
        return generate_poisson_small(rng,mu);
    } else if (mu < 10000.){
        return generate_poisson_large(rng,mu);
    } else if (mu < max_mu) {
        //Approximate by normal
        std::normal_distribution<double> norm_approx(mu,sqrt(mu));
        return norm_approx(rng);
    } else {
        std::ostringstream msg;
        msg<<"Generate Poisson got  mu:"<<mu<<" above maximum safe value:"<<max_mu;
        throw ParameterValueError(msg.str());
    }
}


// template<class RngT>
// double generate_normal(RngT &rng, double mu, double sigma){
//     static double q,r;
//     static unsigned long count=0;
//     double z,s=1;
//     UniformRNG u(-1,1);// ~U[-1,+1]
//     if(count++ % 2) return r*sigma+mu; //postincriment means every odd iteration returns r
//     while(s>=1) {
//         q=u(rng);
//         r=u(rng);
//         s=q*q+r*r;
//     }
//     if(s==0) {
//         q=0;
//         r=0;
//     } else {
//         z=sqrt(-2*log(s)/s);
//         q=q*z;
//         r=r*z;
//     }
//     return q*sigma+mu; //Even iterations compute q&r and return q
// }

} /* namespace mappel */

#endif /* MAPPEL_RNG_H */
