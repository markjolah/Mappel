/** @file test_models1DVariableSigma.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests general to ALL 2D models.
 */

#include "test_mappel.h"
#include "Mappel/Gauss2DMLE.h"
#include "Mappel/Gauss2DMAP.h"
#include "Mappel/Gauss2DsMLE.h"
#include "Mappel/Gauss2DsMAP.h"

template<class Model>
class TestModel2D : public ::testing::Test {
public:    
    Model model;
    TestModel2D() : model(make_model<Model>()) {}
    virtual void SetUp() {
        env->reset_rng();
        model = make_model<Model>();
    }
};

using TypesModel2D = ::testing::Types<mappel::Gauss2DMLE,mappel::Gauss2DMAP,mappel::Gauss2DsMLE,mappel::Gauss2DsMAP>;
TYPED_TEST_CASE(TestModel2D, TypesModel2D);
TYPED_TEST(TestModel2D, size_2D) {
    using Model = TypeParam;
    EXPECT_EQ(Model::num_dim,2);
}

// TYPED_TEST(TestModel2Ds, min_max_sigma_1D) {
//     //"""Check min_sigma and max_sigma get and set properties."""
//     TypeParam &model = this->model;
//     EXPECT_TRUE(std::isfinite(model.get_min_sigma()));
//     EXPECT_TRUE(std::isfinite(model.get_max_sigma()));
//     EXPECT_LT(0,model.global_min_psf_sigma);
//     EXPECT_LE(model.global_min_psf_sigma, model.get_min_sigma()); 
//     EXPECT_LT(model.get_min_sigma(), model.get_max_sigma());
//     EXPECT_LE(model.get_max_sigma(), model.global_max_psf_sigma);
//     //check setter properties
//     double new_min_sigma = model.get_min_sigma();
//     double new_max_sigma = model.get_max_sigma();
//     new_min_sigma *= 0.9;
//     new_max_sigma *= 1.1;
//     model.set_min_sigma(new_min_sigma);
//     model.set_max_sigma(new_max_sigma);
//     EXPECT_EQ(model.get_min_sigma(), new_min_sigma);
//     EXPECT_EQ(model.get_max_sigma(), new_max_sigma);
// }
// 
// TYPED_TEST(TestModel2Ds, min_max_sigma_bounds_1D) {
//     //Check min_sigma and max_sigma are respected in bounds."""
//     TypeParam &model = this->model;
//     auto lbound = model.get_lbound();
//     EXPECT_EQ(model.get_min_sigma(), lbound(3));
//     auto ubound = model.get_ubound();
//     EXPECT_EQ(model.get_max_sigma(), ubound(3));
//     
//     auto new_lbound = lbound;
//     new_lbound(3)*=0.9;
//     model.set_lbound(new_lbound);
//     EXPECT_EQ(new_lbound(3),model.get_lbound()(3));
//     EXPECT_EQ(new_lbound(3),model.get_min_sigma());
// 
//     auto new_ubound = ubound;
//     new_ubound(3)*=1.1;
//     model.set_ubound(new_ubound);
//     EXPECT_EQ(new_ubound(3),model.get_ubound()(3));
//     EXPECT_EQ(new_ubound(3),model.get_max_sigma());
// 
//     new_lbound(3)*=0.9;
//     new_ubound(3)*=1.1;
//     model.set_bounds(new_lbound, new_ubound);
//     EXPECT_EQ(new_lbound(3),model.get_lbound()(3));
//     EXPECT_EQ(new_lbound(3),model.get_min_sigma());
//     EXPECT_EQ(new_ubound(3),model.get_ubound()(3));
//     EXPECT_EQ(new_ubound(3),model.get_max_sigma());
// }
