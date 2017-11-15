/**
 * @file rng.h
 * 
 * @author Mark J. Olah (email mjo\@cs.unm.edu )
 * @date 12-12-2013
 * 
 * @brief Random number generation usign sfmt
 */
#ifndef _RNG_H
#define _RNG_H

#include "util.h"
#include <trng/uniform01_dist.hpp>
#include <trng/normal_dist.hpp>
#include <trng/lcg64_shift.hpp>
#include "parallel_rng/ParallelRngManager.h"

// #include <omp.h>

namespace mappel {

using ParallelRngT = trng::lcg64_shift;
using UnitRNG = trng::uniform01_dist<>;
using NormalRNG = trng::normal_dist<>;

/** @brief Global parallel RNG manager object
 * 
 */
extern parallel_rng::ParallelRngManager<ParallelRngT> rng_manager;


/*template<class RNG>
RNG make_parallel_rng_stream(uint64_t seed)
{
    RNG rng(seed);
    int size=omp_get_num_threads();
    int rank=omp_get_thread_num();
    rng.split(size,rank);
    return rng;
}*/


/** @brief Genrates a single poisson disributed int from distribution with mean mu.
 * @param mu - mean of poisson distribution
 * @param sfmt - A pointer to the SFMT rng state.
 * 
 * Knuth method circa 1969.  Trasformed to work ing log space.  This is linear in mu.  Works ok for small
 * counts.
 */
template<class rng_t>
int generate_poisson(rng_t &rng, double mu);

/** @brief Genrates a single normal disributed number with mean mu and standard deviation sigma
 * 
 * 
 * Uses the Marsaglia algorithm.  As described in Knuth AoCPv2 p122.
 * We use static varibles to count the numer of calls and use both u & v on alternate calls.
 */
// template<class rng_t>
// double generate_normal(rng_t &rng, double mu, double sigma);



template<class rng_t>
int generate_poisson_small(rng_t &rng, double mu)
{
    int k=0;
    double L=exp(-mu);
    if (L>=1.0) return 0;
    UnitRNG u;
    for(double p=1.0; p>L; k++) p*=u(rng);
    return k-1;
}

// "Rejection method PA" from "The Computer Generation of Poisson Random Variables" by A. C. Atkinson
// Journal of the Royal Statistical Society Series C (Applied Statistics) Vol. 28, No. 1. (1979)
// The article is on pages 29-35. The algorithm given here is on page 32.
template<class rng_t>
int generate_poisson_large(rng_t &rng, double mu)
{
    double c = 0.767 - 3.36/mu;
    double beta = arma::datum::pi/sqrt(3.0*mu);
    double alpha = beta*mu;
    double k = log(c) - mu - log(beta);
    UnitRNG uni;

    while(true) {
        double u = 1-uni(rng);//support: (0,1]
        double x = (alpha - log((1.0 - u)/u))/beta;
        int n = (int) floor(x + 0.5);
        if(n<0) continue;
        double v = 1-uni(rng);//support: (0,1]
        double y = alpha - beta*x;
        double temp = 1.0 + exp(y);
        double lhs = y + log(v/(temp*temp));
        double rhs = k + n*log(mu) - lgamma(n+1);
        if (lhs <= rhs)
            return n;
    }
}

template<class rng_t>
int generate_poisson(rng_t &rng, double lambda)
{
    if (lambda<=0. || !std::isfinite(lambda)) {
        return 0;
    } else if (lambda < 30.) {
        return generate_poisson_small(rng,lambda);
    } else {
        return generate_poisson_large(rng,lambda);
    }
}

// template<class rng_t>
// double generate_normal(rng_t &rng, double mu, double sigma){
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


// template<class rng_t>
// double generate_beta(rng_t &rng, const boost::math::beta_distribution<>  &dist)
// {
//     UnitRNG u;
//     return boost::math::quantile(dist,u(rng));
// }
// 
// template<class rng_t>
// double generate_beta(rng_t &rng, double alpha, double beta)
// {
//     UnitRNG u;
//     boost::math::beta_distribution<> d(alpha,beta);
//     return boost::math::quantile(d,u(rng));
// }
// 
// template<class rng_t>
// double generate_gamma(rng_t &rng, const boost::math::gamma_distribution<>  &dist)
// {
//     UnitRNG u;
//     return boost::math::quantile(dist,u(rng));
// }
// 
// template<class rng_t>
// double generate_gamma(rng_t &rng, double shape, double scale)
// {
//     UnitRNG u;
//     boost::math::gamma_distribution<> d(shape,scale);
//     return boost::math::quantile(d,u(rng));
// }

} /* namespace mappel */

/*
namespace trng {
    
    template <typename RealType>
    class beta_dist
    {
    public:
        typedef RealType result_type;
        
        class param_type {
            RealType _a, _b;
        public:
            typedef beta_dist distribution_type;
            explicit param_type(RealType a = 1.0, RealType b = 1.0) : _a(a), _b(b) {}
            RealType a() const { return _a; }
            RealType b() const { return _b; }
            bool operator==(const param_type& other) const { return _a == other._a && _b == other._b;}
            bool operator!=(const param_type& other) const { return !(*this == other); }
        };
        
        explicit beta_dist(RealType a = 2.0, RealType b = 2.0): a_dist(a), b_dist(b) { }
        explicit beta_dist(const param_type& param): a_dist(param.a()), b_dist(param.b()) { }
        
        void reset() { }
        
        param_type param() const { return param_type(a(), b()); }
        
        void param(const param_type& param)
        {
            a_dist = gamma_dist_type(param.a());
            b_dist = gamma_dist_type(param.b());
        }
        
        void set_params(double alpha, double beta)
        {
            a_dist = gamma_dist_type(alpha);
            b_dist = gamma_dist_type(beta);
        }
        
        template <typename URNG>
        result_type operator()(URNG& engine) { return generate(engine, a_dist, b_dist); }
        
        template <typename URNG>
        result_type operator()(URNG& engine, const param_type& param)
        {
            gamma_dist_type tmp_a_dist(param.a());
            gamma_dist_type tmp_b_dist(param.b());
            return generate(engine, tmp_a_dist, tmp_b_dist);
        }
        
        result_type min() const { return 0.0; }
        result_type max() const { return 1.0; }
        
        result_type a() const { return a_dist.alpha(); }
        result_type b() const { return b_dist.alpha(); }
        
        void a(RealType aparam) { a_dist = gamma_dist_type(aparam); }
        void b(RealType bparam) { b_dist = gamma_dist_type(bparam); }
        
        bool operator==(const beta_dist<result_type>& other) const
        {
            return param() == other.param() && a_dist == other.a_dist && b_dist == other.b_dist;
        }
        
        bool operator!=(const beta_dist<result_type>& other) const { return !(*this == other);}
        
    private:
        typedef std::gamma_distribution<result_type> gamma_dist_type;
        
        gamma_dist_type a_dist, b_dist;
        
        template <typename URNG>
        result_type generate(URNG& engine, gamma_dist_type& x_gamma, gamma_dist_type& y_gamma)
        {
            result_type x=x_gamma(engine);
            result_type y=y_gamma(engine);
            if (x+y==0.) return 0.;
            else return x/(x+y);
        }
    };*/
    
// } /* namespace trng */

#endif /* _RNG_H */
