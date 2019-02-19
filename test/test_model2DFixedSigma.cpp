/** @file test_models2DFixedSigma.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests specific to 2D models with a fixed sigma
 */

#include "test_mappel.h"
#include "Mappel/Gauss2DMLE.h"
#include "Mappel/Gauss2DMAP.h"

template<class Model>
class TestModel2DFixedSigma : public ::testing::Test {
public:    
    Model model;
    TestModel2DFixedSigma() : model(make_model<Model>()) {}
    virtual void SetUp() {
        env->reset_rng();
        model = make_model<Model>();
    }
};

using TypesModel2DFixedSigma = ::testing::Types<mappel::Gauss2DMLE,mappel::Gauss2DMAP> ;
TYPED_TEST_CASE(TestModel2DFixedSigma, TypesModel2DFixedSigma);


template<class Model>
void check_sum_model_consitency(const Model &model)
{
    auto x_model = model.debug_internal_sum_model_x();
    auto y_model = model.debug_internal_sum_model_y();
    EXPECT_EQ(x_model.get_size(), model.get_size()(0));
    EXPECT_EQ(y_model.get_size(), model.get_size()(1));
    EXPECT_EQ(x_model.get_psf_sigma(), model.get_psf_sigma()(0));
    EXPECT_EQ(y_model.get_psf_sigma(), model.get_psf_sigma()(1));
    auto x_prior = x_model.get_prior();
    //Test prior
}


// TYPED_TEST(TestModel2DFixedSigma, psf_sigma) {
// //     TypeParam &model = this->model;
// //     EXPECT_GE(model.get_psf_sigma(),model.global_min_psf_sigma);
// //     EXPECT_LE(model.get_psf_sigma(),model.global_max_psf_sigma);
// //     double new_sigma = model.get_psf_sigma();
// //     new_sigma+=0.01;
// //     model.set_psf_sigma(new_sigma);
// //     EXPECT_EQ(model.get_psf_sigma(),new_sigma); 
// }
