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


TYPED_TEST(TestModel2D, size) {
    TypeParam &model = this->model;
    auto size = model.get_size();
    for(int dim=0;dim<2;dim++){
        EXPECT_LE(model.global_min_size, size(dim))<<"Dim: "<<dim;
        EXPECT_GE(model.global_max_size, size(dim))<<"Dim: "<<dim;
    }
}

TYPED_TEST(TestModel2D, set_size) {
    TypeParam &model = this->model;
    auto size = model.get_size();
    size(0) = model.global_min_size-1
    EXPECT_THROW(model.set_size(size), mappel::ParameterValueError);
    size(0) = model.global_max_size+1
    EXPECT_THROW(model.set_size(size), mappel::ParameterValueError);
    auto new_size = model.get_size();
    new_size+=1;
    model.set_size(new_size);
    for(int dim=0;dim<2;dim++) EXPECT_EQ(model.get_size()(dim),new_size(dim))<<"Dim: "<<dim;
}
