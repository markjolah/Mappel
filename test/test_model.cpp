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
    Model model2;
    TestModel() : model(make_model<Model>()),
                  model2(make_model<Model>()) { }
    virtual void SetUp() {
        env->reset_rng();
//         std::cout<<"Model: "<<model<<" Prior["<<&model.get_prior()<<"]:"<<model.get_prior()<<"\n ubound:"<<model.get_ubound().t()<<" prior.ubound:"<<model.get_prior().ubound().t()<<std::endl;
        model = make_model<Model>();
        model2 = make_model<Model>();
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

TYPED_TEST(TestModel, get_num_hyoerparams) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    ASSERT_GE(N,1);
    auto p = model.get_prior();
    ASSERT_EQ(p.num_params(), N);
}

TYPED_TEST(TestModel, get_hyperparams) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    auto hps = model.get_hyperparams();
    ASSERT_EQ(hps.n_elem, N);
}

TYPED_TEST(TestModel, set_hyperparams) {
    TypeParam &model1 = this->model;
    TypeParam &model2 = this->model2;
    auto N1 = model1.get_num_hyperparams();
    auto N2 = model1.get_num_hyperparams();
    auto hp1 = model1.get_hyperparams();
    auto hp2 = model2.get_hyperparams();
    ASSERT_EQ(N1, N2);
    ASSERT_EQ(hp1.n_elem, N1);
    ASSERT_EQ(hp2.n_elem, N2);
    model1.set_hyperparams(hp2);
    model2.set_hyperparams(hp1);
    auto hp11 = model1.get_hyperparams();
    auto hp22 = model2.get_hyperparams();
    EXPECT_TRUE(arma::all(hp1==hp22));
    EXPECT_TRUE(arma::all(hp2==hp11));
}

TYPED_TEST(TestModel, has_hyperparam) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    auto names = model.get_hyperparam_names();
    ASSERT_EQ(names.size(), N);
    for(auto &n: names) {
        EXPECT_TRUE(model.has_hyperparam(n));
        EXPECT_FALSE(model.has_hyperparam(n+"_FOO"));
    }
}

TYPED_TEST(TestModel, get_set_hyperparam_value) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    auto names = model.get_hyperparam_names();
    auto hps = model.get_hyperparams();
    ASSERT_EQ(hps.n_elem, N);
    ASSERT_EQ(names.size(), N);
    for(IdxT i=0;i<N;i++) {
        auto n = names[i];
        auto v = hps(i);
        EXPECT_TRUE(model.has_hyperparam(n));
        EXPECT_EQ(model.get_hyperparam_index(n),i);
        EXPECT_EQ(model.get_hyperparam_value(n),v);
        auto v2 = v*0.99999;
        model.set_hyperparam_value(n,v2);
        EXPECT_EQ(model.get_hyperparam_value(n),v2);
    }
}

TYPED_TEST(TestModel, rename_hyperparam) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    auto names = model.get_hyperparam_names();
    auto hps = model.get_hyperparams();
    ASSERT_EQ(hps.n_elem, N);
    ASSERT_EQ(names.size(), N);
    for(IdxT i=0;i<N;i++) {
        auto n = names[i];
        auto v = hps(i);
        EXPECT_TRUE(model.has_hyperparam(n));
        EXPECT_EQ(model.get_hyperparam_index(n),i);
        EXPECT_EQ(model.get_hyperparam_value(n),v);
        auto n2 = n+'\'';
        model.rename_hyperparam(n,n2);
        EXPECT_TRUE(model.has_hyperparam(n2));
        EXPECT_FALSE(model.has_hyperparam(n));
        EXPECT_EQ(model.get_hyperparam_index(n2),i);
        EXPECT_EQ(model.get_hyperparam_value(n2),v);

        EXPECT_THROW(model.get_hyperparam_index(n),prior_hessian::ParameterNameError);
        EXPECT_THROW(model.get_hyperparam_value(n),prior_hessian::ParameterNameError);

        model.rename_hyperparam(n2,n);
        EXPECT_TRUE(model.has_hyperparam(n));
        EXPECT_FALSE(model.has_hyperparam(n2));
        EXPECT_EQ(model.get_hyperparam_index(n),i);
        EXPECT_EQ(model.get_hyperparam_value(n),v);

        EXPECT_THROW(model.get_hyperparam_index(n2),prior_hessian::ParameterNameError);
        EXPECT_THROW(model.get_hyperparam_value(n2),prior_hessian::ParameterNameError);
    }
}

TYPED_TEST(TestModel, get_set_param_names) {
    TypeParam &model = this->model;
    auto N = model.get_num_params();
    auto names = model.get_param_names();
    ASSERT_EQ(names.size(),N);
    auto new_names = model.get_param_names();
    for(auto &n: new_names) n+='\'';

    model.set_param_names(new_names);
    auto names2 = model.get_param_names();
    ASSERT_EQ(names2.size(),N);
    for(IdxT i=0; i<N; i++) {
        EXPECT_NE(names[i],new_names[i]);
        EXPECT_EQ(names2[i],new_names[i]);
    }
}


TYPED_TEST(TestModel, get_set_hyperparam_names) {
    TypeParam &model = this->model;
    auto N = model.get_num_hyperparams();
    auto names = model.get_hyperparam_names();
    ASSERT_EQ(names.size(),N);
    auto new_names = model.get_hyperparam_names();
    for(auto &n: new_names) n+='\'';

    model.set_hyperparam_names(new_names);
    auto names2 = model.get_hyperparam_names();
    ASSERT_EQ(names2.size(),N);
    for(IdxT i=0; i<N; i++) {
        EXPECT_NE(names[i],new_names[i]);
        EXPECT_EQ(names2[i],new_names[i]);
    }
}

TYPED_TEST(TestModel, sample_prior_inbounds) {
    TypeParam &model = this->model;
    auto N = model.get_num_params();
    for(IdxT n=0; n<Nsample; n++){
        auto theta = model.sample_prior();
        ASSERT_EQ(theta.n_elem, N);
        EXPECT_TRUE(model.theta_in_bounds(theta));
    }
}

//Check model and prior lbound are the same
TYPED_TEST(TestModel, lbound_prior) {
    TypeParam &model = this->model;
    auto lbound = model.get_lbound();
    auto prior = model.get_prior();
    auto plbound = prior.lbound();
    ASSERT_EQ(lbound.n_elem,model.get_num_params());
    ASSERT_EQ(plbound.n_elem,model.get_num_params());
    for(IdxT i=0;i<model.get_num_params();i++)
        EXPECT_EQ(lbound(i),plbound(i));
}

//Check model and prior ubound are the same
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

TYPED_TEST(TestModel, lbound_ubound_ordering) {
    TypeParam &model = this->model;
    auto N = model.get_num_params();
    auto lbound = model.get_lbound();
    auto ubound = model.get_ubound();
    ASSERT_EQ(lbound.n_elem, N);
    ASSERT_EQ(ubound.n_elem, N);
    EXPECT_TRUE(arma::all(lbound < ubound));
}



