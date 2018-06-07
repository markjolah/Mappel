/** @file test_mappel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Common Mappel testing environment
 */
#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"
#include "Mappel/Gauss1DModel.h"
#include "Mappel/Gauss1DsModel.h"

/* Globals */
extern test_helper::RngEnvironment *env;

/* Factory functions */
template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss1DModel,Model>::value,Model>::type
make_model()
{
    int size = env->sample_integer(4,40);
    double psf_sigma = size*env->sample_exponential(1.5);
    std::cout<<"1DModel Generated[size:"<<size<<", psf_sigma:"<<psf_sigma<<std::endl;
    return Model(size,psf_sigma);
}

template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss1DsModel,Model>::value,Model>::type
make_model()
{
    int size = env->sample_integer(4,40);
    double min_sigma = size*env->sample_real(0.1,.3);
    double max_sigma = env->sample_real(min_sigma*1.2,min_sigma*6);
    std::cout<<"1DsModel Generated[size:"<<size<<", sigma:["<<min_sigma<<","<<max_sigma<<"]"<<std::endl;
    return Model(size,min_sigma,max_sigma);
}

