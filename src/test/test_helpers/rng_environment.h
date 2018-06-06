/** @file rng_environment.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief An environment for use in googletest that enables repeatable randomized testing
 */
#include<random>
#include<iostream>
#include "gtest/gtest.h"

namespace test_helper {

class RngEnvironment : public ::testing::Environment 
{
    using SeedT = uint64_t;
    using RngT = std::mt19937_64;
    static const SeedT MAX_SEED = 9999; //Limit seed size to make typing it in on command line as gtest exe argument easier
    SeedT seed = 0;
    RngT rng;
public:
    SeedT get_seed() const {return seed;}
    
    void set_seed(SeedT _seed)
    {
        seed = _seed;
    }
    
    void set_seed()
    {
        //Generate a small human-typeable seed value.  This will give us good enough coverage and make
        //it easy to enter seeds from the command line
        std::random_device rng;
        std::uniform_int_distribution<SeedT> seed_dist(0,MAX_SEED);
        seed = seed_dist(rng);
    }
    
    void SetUp() const
    {
        ::testing::Test::RecordProperty("rng_seed",seed);
        std::cout<<">>>>>>>>>>>> To Repeat Use SEED: "<<seed<<"\n";
    }

    //Use saved seed to reset the RNG.  Typically called before each test to make them independent of ordering.
    void reset_rng()
    {
        rng.seed(seed);
    }
    
    double sample_real(double a, double b) 
    {
        std::uniform_real_distribution<double> d(a,b);
        return d(rng);
    }

    template<class IntT>
    IntT sample_integer(IntT a, IntT b) 
    {
        std::uniform_int_distribution<IntT> d(a,b);
        return d(rng);
    }

    double sample_normal(double mean, double sigma) 
    {
        std::normal_distribution<double> d(mean,sigma);
        return d(rng);
    }
    
    double sample_exponential(double lambda) 
    {
        std::exponential_distribution<double> d(lambda);
        return d(rng);
    }

    double sample_gamma(double alpha, double beta) 
    {
        std::gamma_distribution<double> d(alpha,beta);
        return d(rng);
    }

    RngT& get_rng()
    {
        return rng;
    }
};

} /* namespace test_helper */
