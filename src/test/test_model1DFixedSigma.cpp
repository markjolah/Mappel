/** @file test_models1DFixedSigma.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests specific to 1D models with a fixed sigma
 */

#include "test_mappel.h"
#include "Mappel/Gauss1DMLE.h"
#include "Mappel/Gauss1DMAP.h"

template<class Model>
class TestModel1DFixedSigma : public ::testing::Test {
public:    
    Model model;
    TestModel1DFixedSigma() : model(make_model<Model>()) {}
    virtual void SetUp() {
        env->reset_rng();
        model = make_model<Model>();
    }
};

using TypesModel1DFixedSigma = ::testing::Types<mappel::Gauss1DMLE,mappel::Gauss1DMAP> ;
TYPED_TEST_CASE(TestModel1DFixedSigma, TypesModel1DFixedSigma);


TYPED_TEST(TestModel1DFixedSigma, psf_sigma) {
    TypeParam &model = this->model;
    EXPECT_GE(model.get_psf_sigma(),model.global_min_psf_sigma);
    EXPECT_LE(model.get_psf_sigma(),model.global_max_psf_sigma);
    double new_sigma = model.get_psf_sigma();
    new_sigma+=0.01;
    model.set_psf_sigma(new_sigma);
    EXPECT_EQ(model.get_psf_sigma(),new_sigma); 
}
