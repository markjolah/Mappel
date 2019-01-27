/** @file test_mappel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Common Mappel testing environment
 */
#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"
#include "Mappel/Gauss1DModel.h"
#include "Mappel/Gauss1DsModel.h"
#include "Mappel/Gauss2DModel.h"
#include "Mappel/Gauss2DsModel.h"

using mappel::IdxT;
using mappel::VecT;
using mappel::MatT;

/* Globals */
extern test_helper::RngEnvironment *env;
extern IdxT Nsample;



/* Factory functions */
template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss1DModel,Model>::value,Model>::type
make_model()
{
    int size = env->sample_integer(4,40);
    double psf_sigma = size*env->sample_exponential(1.5);
//     std::cout<<"1DModel Generated[size:"<<size<<", psf_sigma:"<<psf_sigma<<std::endl;
    return Model(size,psf_sigma);
}

template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss1DsModel,Model>::value,Model>::type
make_model()
{
    int size = env->sample_integer(4,40);
    double min_sigma = size*env->sample_real(0.1,.3);
    double max_sigma = env->sample_real(min_sigma*1.2,min_sigma*6);
//     std::cout<<"1DsModel Generated[size:"<<size<<", sigma:["<<min_sigma<<","<<max_sigma<<"]"<<std::endl;
    return Model(size,min_sigma,max_sigma);
}

template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss2DModel,Model>::value,Model>::type
make_model()
{
    int sizex = env->sample_integer(4,40);
    int sizey = env->sample_integer(std::max(4,sizex-5),std::min(sizex+5,40));
    double sigmax = sizex*env->sample_real(0.1,.3);
    double sigmay = sizey*env->sample_real(0.1,.3);
    typename Model::ImageSizeT size = {static_cast<typename Model::ImageCoordT>(sizex),static_cast<typename Model::ImageCoordT>(sizey)};
    VecT psf_sigma={sigmax,sigmay};
//     std::cout<<"2DModel Generated[size:"<<sizex<<","<<sizey<<"], psf_sigma:["<<sigmax<<","<<sigmay<<"]"<<std::endl;
    return Model(size, psf_sigma);
}

template<class Model>
typename std::enable_if<std::is_base_of<mappel::Gauss2DsModel,Model>::value,Model>::type
make_model()
{
    int sizex = env->sample_integer(4,40);
    int sizey = env->sample_integer(std::max(4,sizex-5),std::min(sizex+5,40));
    double sigmax = sizex*env->sample_real(0.1,.3);
    double sigmay = sizey*env->sample_real(0.1,.3);
    double max_sigma = env->sample_real(2.2,6);
    typename Model::ImageSizeT size = {static_cast<typename Model::ImageCoordT>(sizex),static_cast<typename Model::ImageCoordT>(sizey)};
    VecT psf_sigma={sigmax,sigmay};
//     std::cout<<"2DsModel Generated[size:"<<sizex<<","<<sizey<<"], psf_sigma:["<<sigmax<<","<<sigmay<<"] max_sigma:"<<max_sigma<<std::endl;
    return Model(size,psf_sigma,max_sigma);
}

