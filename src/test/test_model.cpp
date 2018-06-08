/** @file test_models.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests generic to any model
 */

#include "test_mappel.h"
#include "Mappel/Gauss1DMLE.h"
#include "Mappel/Gauss1DMAP.h"
#include "Mappel/Gauss1DsMLE.h"
#include "Mappel/Gauss1DsMAP.h"
#include "Mappel/Gauss2DMLE.h"
#include "Mappel/Gauss2DMAP.h"
#include "Mappel/Gauss2DsMLE.h"
#include "Mappel/Gauss2DsMAP.h"
template<class Model>
class TestModel : public ::testing::Test {
public:    
    Model model;
    TestModel() : model(make_model<Model>()) {}
    virtual void SetUp() {
        env->reset_rng();
//         std::cout<<"Model: "<<model<<" Prior["<<&model.get_prior()<<"]:"<<model.get_prior()<<"\n ubound:"<<model.get_ubound().t()<<" prior.ubound:"<<model.get_prior().ubound().t()<<std::endl;
        model = make_model<Model>();
//         std::cout<<"Model: "<<model<<" Prior["<<&model.get_prior()<<"]:"<<model.get_prior()<<"\n ubound:"<<model.get_ubound().t()<<" prior.ubound:"<<model.get_prior().ubound().t()<<std::endl;
    }
};

using TypesModel = ::testing::Types<mappel::Gauss1DMLE,mappel::Gauss1DMAP,mappel::Gauss1DsMLE,mappel::Gauss1DsMAP,
                                    mappel::Gauss2DMLE,mappel::Gauss2DMAP,mappel::Gauss2DsMLE,mappel::Gauss2DsMAP>;
TYPED_TEST_CASE(TestModel, TypesModel);


TYPED_TEST(TestModel, prior_get_set) {
    TypeParam &model = this->model;
    auto ubound = model.get_ubound();
    const auto& prior = model.get_prior();
    const auto& prior2 = model.get_prior();
    ASSERT_EQ(&prior,&prior2);
    auto pubound = prior.ubound();
    auto pubound2 = prior2.ubound();
    auto ubound2 = model.get_ubound();
    ASSERT_EQ(ubound.n_elem,model.get_num_params());
    ASSERT_EQ(pubound.n_elem,model.get_num_params());
    ASSERT_EQ(pubound2.n_elem,model.get_num_params());
    for(IdxT i=0;i<model.get_num_params();i++) {
        EXPECT_EQ(pubound2(i),pubound(i)) <<"i: "<<i;
        EXPECT_EQ(ubound(i),pubound(i)) <<"i: "<<i;
        EXPECT_EQ(ubound2(i),pubound(i)) <<"i: "<<i;
    }
}

TYPED_TEST(TestModel, lbound_prior) {
    TypeParam &model = this->model;
    auto lbound = model.get_lbound();
    auto prior = model.get_prior();
//     std::cout<<"Prior: "<<prior<<std::endl;
    auto plbound = prior.lbound();
    ASSERT_EQ(lbound.n_elem,model.get_num_params());
    ASSERT_EQ(plbound.n_elem,model.get_num_params());
    for(IdxT i=0;i<model.get_num_params();i++)
        EXPECT_EQ(lbound(i),plbound(i));
}

TYPED_TEST(TestModel, ubound_prior) {
    TypeParam &model = this->model;
    auto ubound = model.get_ubound();
    const auto& prior = model.get_prior();
    auto pubound = prior.ubound();
    const auto& prior2 = model.get_prior();
    ASSERT_EQ(&prior,&prior2);
    ASSERT_EQ(ubound.n_elem,model.get_num_params());
    ASSERT_EQ(pubound.n_elem,model.get_num_params());
    for(IdxT i=0;i<model.get_num_params();i++)
        EXPECT_EQ(ubound(i),pubound(i));
}
