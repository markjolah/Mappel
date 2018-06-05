/** @file test_mappel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Common Mappel testing environment
 */
#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"


/* Globals */
extern test_helper::RngEnvironment *env;

/* Factory functions */
// template<class Dist> 
// Dist make_dist();
// template<> prior_hessian::NormalDist make_dist();


/* Type parameterized test fixtures */
// template<class Dist>
// class UnivariateDistTest : public ::testing::Test {
// public:    
//     Dist dist{0,1,"x"};
//     virtual void SetUp() {
//         env->reset_rng();
//         dist = make_dist<Dist>();
//     }
// };
