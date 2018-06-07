/** @file test_models1D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests specific to 1D models
 */

#include "test_mappel.h"
#include "Mappel/Gauss1DMLE.h"
#include "Mappel/Gauss1DMAP.h"
#include "Mappel/Gauss1DsMLE.h"
#include "Mappel/Gauss1DsMAP.h"

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

using TypesModel1D = ::testing::Types<mappel::Gauss1DMLE,mappel::Gauss1DMAP,
                                      mappel::Gauss1DsMLE,mappel::Gauss1DsMAP>;
TYPED_TEST_CASE(TestModel1D, TypesModel1D);

TYPED_TEST(TestModel1D, num_dim) {
    EXPECT_EQ(this->model.num_dim,1)<<"1D Model";
}

TYPED_TEST(TestModel1D, size) {
    EXPECT_LE(this->model.global_min_size, this->model.get_size());
    EXPECT_GE(this->model.global_max_size, this->model.get_size());
}

TYPED_TEST(TestModel1D, set_size) {
    TypeParam &model = this->model;
    EXPECT_THROW(model.set_size(model.global_min_size-1), mappel::ParameterValueError);
    EXPECT_THROW(model.set_size(model.global_max_size+1), mappel::ParameterValueError);
    auto new_size = model.get_size();
    new_size+=1;
    model.set_size(new_size);
    EXPECT_EQ(model.get_size(),new_size);
}
