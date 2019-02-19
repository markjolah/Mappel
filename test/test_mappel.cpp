/** @file test_mappel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Main google test for Mappel C++ interface
 */
#include "test_mappel.h"
#include "BacktraceException/BacktraceException.h"

/* Globals */
test_helper::RngEnvironment *env = new test_helper::RngEnvironment; //Googletest wants to free env, so we need to appease its demands or face segfaults.
IdxT Nsample=100;


int main(int argc, char **argv)
{
    if(argc>2 && !strncmp("--seed",argv[1],6)){
        char* end;
        auto seed = strtoull(argv[2],&end,0);
        env->set_seed(seed);
        //Pass on seed to G-test as command line argument
        const int N=30;
        char buf[N];
        snprintf(buf,N,"--gtest_random_seed=%llu",seed);
        argv[2] = buf;
        argc--; argv++;
    } else {
        env->set_seed();
    }
    ::testing::AddGlobalTestEnvironment(env);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
