/** @file test_mappel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Main google test for Mappel C++ interface
 */
#include "test_mappel.h"
#include "BacktraceException/BacktraceException.h"

/* Globals */
test_helper::RngEnvironment *env = new test_helper::RngEnvironment; //Googletest wants to free env, so we need to appease its demands or face segfaults.

int main(int argc, char **argv) 
{
    if(argc>1) {
        char* end;
        env->set_seed(strtoull(argv[0],&end,0));
    } else {
        env->set_seed();
    }
    
    backtrace_exception::disable_backtraces();
    
    ::testing::AddGlobalTestEnvironment(env);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
