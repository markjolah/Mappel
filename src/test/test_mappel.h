/** @file test_mappel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Common Mappel testing environment
 */
#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"
#include "Mappel/Gauss1DModel.h"

/* Globals */
extern test_helper::RngEnvironment *env;

/* Factory functions */
template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss1DModel,Model>::value,Model>::type
make_model()
{
    int size = env->sample_integer(4,40);
    double psf_sigma = size*env->sample_exponential(1.5);
    std::cout<<"1DModel Generated[size:"<<size<<", psf_sigma:"<<psf_sigma<<"\n";
    return Model(size,psf_sigma);
}


/* Type parameterized test fixtures */
template<class Model>
class TestModel1D : public ::testing::Test {
public:    
    Model model;
    TestModel1D() : model(make_model<Model>()) {}
    virtual void SetUp() {
        env->reset_rng();
        model = make_model<Model>();
    }
};
