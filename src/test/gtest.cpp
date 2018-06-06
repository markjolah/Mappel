#include <tuple>
#include <armadillo>

#include "gtest/gtest.h"

#include "Mappel/Gauss2DMLE.h"
#include "Mappel/Gauss2DMAP.h"
// #include "Mappel/Gauss2DsMLE.h"
// #include "Mappel/Gauss2DsMAP.h"
// #include "Mappel/Blink2DsMAP.h"
// 
// #include "Mappel/GaussHSMAP.h"
// #include "Mappel/GaussHSsMAP.h"
// #include "Mappel/BlinkHSsMAP.h"

const double ftol=4E-5;

using testing::TestWithParam;
using namespace arma;
using namespace mappel;
namespace { //Continues to end of file

using std::tuple;
using std::get;
using std::vector;
using std::make_tuple;
using std::make_shared;
using std::shared_ptr;


template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value>::type
make_models(vector<shared_ptr<Model>> &models)
{
    IMatT sizes(3,4);
    sizes.col(0)=IVecT({4,7,5});
    sizes.col(1)=IVecT({13,10,10});
    sizes.col(2)=IVecT({7,7,9});
    sizes.col(3)=IVecT({12,15,12});

    MatT sigma(3,4);
    sigma.col(0)=VecT({0.3,1.05,1.});
    sigma.col(1)=VecT({1.45,1.3141,1.2});
    sigma.col(2)=VecT({1.0,1.0,1.2});
    sigma.col(3)=VecT({1.8,2.0,1.5});

    for(unsigned i=0; i<sizes.n_cols; i++)
        models.push_back(make_shared<Model>(Model(sizes.col(i),sigma.col(i))));
}


template<class Model>
typename std::enable_if<std::is_base_of<PointEmitter2DModel,Model>::value>::type
make_models(vector<shared_ptr<Model>> &models)
{
    const vector<tuple<int,int,double,double>> params={make_tuple(4,4,0.3,0.3),
                                            make_tuple(8,8,1.,1.0),
                                            make_tuple(9,9,1.3,1.3),
                                            make_tuple(16,16,2.0,2.0),
                                            make_tuple(21,21,3.141,3.141)};
    for(auto i=params.cbegin(); i!=params.cend(); ++i){
        IVecT sizes(2);
        sizes<<get<0>(*i)<<get<1>(*i);
        VecT psf_sigma={get<2>(*i),get<3>(*i)};
    
        models.push_back(make_shared<Model>(Model(sizes,psf_sigma)));
    }
}




template<class Model>
class PointEmitterModelTest : public ::testing::Test
{
protected:
        RNG rng;
        int num_tests=25;
        vector<shared_ptr<Model>> models;
        PointEmitterModelTest()
            : rng(RNG(make_seed()))
        {
            make_models(models);
//             for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
//                 auto model=*model_it;
//                 std::cout<<"Model:"<<*model<<std::endl;
//             }
        }
};


/* Delcare the model types to test on */
typedef ::testing::Types<Gauss2DMLE,Gauss2DMAP,Gauss2DsMLE,Gauss2DsMAP,Blink2DsMAP,GaussHSMAP,GaussHSsMAP,BlinkHSsMAP> ModelTypes;
TYPED_TEST_CASE(PointEmitterModelTest, ModelTypes);


template<class ArmaVec>
void test_vec_eq(const ArmaVec &p1, const ArmaVec &p2, const char *msg)
{
    EXPECT_EQ(p1.n_elem, p2.n_elem)<<"Vecs have unqual length";
    for(unsigned k=0; k<p1.n_elem; k++)
        EXPECT_FLOAT_EQ(p1(k), p2(k))<<msg<<"Vecs unequal at:"<<k;
}

template<unsigned N>
void test_vec_eq(const Col<double>::fixed<N> &p1, const Col<double>::fixed<N> &p2, const char *msg)
{
    for(unsigned k=0; k<N; k++)
        EXPECT_FLOAT_EQ(p1(k), p2(k))<<msg<<"Vecs unequal at:"<<k;
}

template<class ArmaMat>
void test_mat_eq(const ArmaMat &p1, const ArmaMat &p2, const char *msg)
{
    EXPECT_EQ(p1.n_rows, p2.n_rows)<<"Mats have unqual rows";
    EXPECT_EQ(p1.n_cols, p2.n_cols)<<"Mats have unqual cols";
    for(unsigned j=0; j<p1.n_cols; j++)
        for(unsigned i=0; i<p1.n_rows; i++) //col major for armadillo
            EXPECT_FLOAT_EQ(p1(i,j), p2(i,j))<<msg<<"Mat unequal at:("<<i<<","<<j<<")";
}

template<unsigned N>
void test_mat_eq(const Mat<double>::fixed<N,N> &p1, const Mat<double>::fixed<N,N> &p2, const char *msg)
{
    for(unsigned j=0; j<N; j++)
        for(unsigned i=0; i<N; i++) //col major for armadillo
            EXPECT_FLOAT_EQ(p1(i,j), p2(i,j))<<msg<<"Mat unequal at:("<<i<<","<<j<<")";
}


template<class ArmaMat>
void test_image_eq(const ArmaMat &im1, const ArmaMat &im2, const char *msg)
{
    EXPECT_EQ(im1.n_rows, im2.n_rows)<<"Images have unqual rows";
    EXPECT_EQ(im1.n_cols, im2.n_cols)<<"Images have unqual cols";
    EXPECT_EQ(im1.n_cols, im1.n_rows)<<"Images not square";
    for(unsigned j=0; j<im1.n_cols; j++)
        for(unsigned i=0; i<im1.n_rows; i++) //col major for armadillo
            EXPECT_FLOAT_EQ(im1(i,j), im2(i,j))<<msg<<"Images unequal at:("<<i<<","<<j<<")";
}

template<class ArmaCube>
void test_HS_image_eq(const ArmaCube &im1, const ArmaCube &im2, const char *msg)
{
    EXPECT_EQ(im1.n_rows, im2.n_rows)<<"Images have unqual rows";
    EXPECT_EQ(im1.n_cols, im2.n_cols)<<"Images have unqual cols";
    EXPECT_EQ(im1.n_slices, im2.n_slices)<<"Images have unqual slices";
    for(unsigned k=0; k<im1.n_slices; k++)
        for(unsigned j=0; j<im1.n_cols; j++)
            for(unsigned i=0; i<im1.n_rows; i++) //col major for armadillo
                EXPECT_FLOAT_EQ(im1(i,j,k), im2(i,j,k))<<msg<<"Images unequal at:("<<i<<","<<j<<","<<k<<")";
}


template<class Arma>
bool vec_near(const Arma &p1, const Arma &p2)
{
    static double tol=1e-10;
    for(unsigned n=0; n<p1.n_elem; n++) if ( fabs(p1(n)-p2(n))>tol) return false;
    return true;
}

template<class Arma>
bool vec_ne(const Arma &p1, const Arma &p2)
{
    for(unsigned n=0; n<p1.n_elem; n++) if (p1(n)!=p2(n)) return true;
    return false;
}


template<class Arma>
bool vec_all_ne(const Arma &p1, const Arma &p2)
{
    for(unsigned n=0; n<p1.n_elem; n++) if (p1(n)==p2(n)) return false;
    return true;
}

template<class Arma>
bool vec_all_nonneg(const Arma &p)
{
    for(unsigned n=0; n<p.n_elem; n++) if (p(n)<0) return false;
    return true;
}



template<class Model>
void check_bounds(const Model &model, const typename Model::ParamT &theta_orig, const char*msg="")
{
    typename Model::ParamT theta=theta_orig;
    model.bound_theta(theta);
    for(unsigned  i=0;i<theta.n_elem;i++){
        EXPECT_EQ(theta(i), theta_orig(i)) << "Param:"<<i<<":"<<model.param_names[i]<<" bounds violation: "<<msg;
    }
    EXPECT_TRUE( model.theta_in_bounds(theta))<<"Theta not in bounds!";
}


TYPED_TEST(PointEmitterModelTest, SamplePrior)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        for(int i=0; i < this->num_tests; i++) {
            auto theta=model->sample_prior(this->rng);
            check_bounds<TypeParam>(*model, theta);
        }
    }
}

TYPED_TEST(PointEmitterModelTest, BoundTheta)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        for(int i=0;i<this->num_tests;i++){
            auto theta=model->sample_prior(this->rng);
            for(int a=-3;a<3;a++) for(int b=-3;b<3;b++) {
                theta=theta*a+b;
                model->bound_theta(theta);
                check_bounds<TypeParam>(*model, theta);
            }
        }
    }
}

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitter2DModel,Model>::value>::type
check_model_image(const Model &model, const typename Model::Stencil &s, const typename Model::ImageT &im)
{
    double mv_im,mv;
    EXPECT_PRED1(&::vec_all_nonneg<typename Model::ImageT>,im)<<"Image has negative values";
    for(int i=0;i<model.size(0);i++)  for(int j=0;j<model.size(1);j++)  {
        mv=model.model_value(i,j,s);
        mv_im=im(j,i);
        EXPECT_FLOAT_EQ(mv, mv_im)<<"Model Value unqual at ["<<i<<","<<j<<"]";
    }
}

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value>::type
check_model_image(const Model &model, const typename Model::Stencil &s, const typename Model::ImageT &im)
{
    double mv_im,mv;
    EXPECT_PRED1(&::vec_all_nonneg<typename Model::ImageT>,im)<<"Image has negative values";
    for(int k=0; k<model.size(2); k++) for(int j=0; j<model.size(1); j++) for(int i=0; i<model.size(0); i++) {
        mv=model.model_value(i,j,k,s);
        mv_im=im(i,j,k);
        EXPECT_FLOAT_EQ(mv, mv_im)<<"Model Value unqual at ["<<i<<","<<j<<"]";
    }
}




TYPED_TEST(PointEmitterModelTest, ModelImage)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        for(int n=0;n<this->num_tests;n++){
            auto theta=model->sample_prior(this->rng);
            auto s=model->make_stencil(theta);
            //Check model_image from theta
            auto im=model_image(*model,theta);
            check_model_image(*model,s,im);

            //Check model_image from stencil
            im=model_image(*model,s);
            check_model_image(*model,s,im);
        }
    }
}


// Stack testing
TYPED_TEST(PointEmitterModelTest, SampleThetaPriorStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto theta=model->make_param_stack(this->num_tests);
        sample_prior_stack(*model, theta);
        for(int i=0;i<this->num_tests;i++) {
            check_bounds<TypeParam>(*model,theta.col(i));
            if (i>0) {
                EXPECT_PRED2(&::vec_ne<VecT>, theta.col(i), theta.col(i-1))<<
                                            "successive theta samples identical";
                EXPECT_PRED2(&::vec_ne<VecT>, theta.col(i), theta.col(i-1))<<
                            "successive theta samples identical individual values";
            }
        }
    }
}

TYPED_TEST(PointEmitterModelTest, ModelImageStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto im_stack=model->make_image_stack(this->num_tests);
        auto theta_stack=model->make_param_stack(this->num_tests);
        sample_prior_stack(*model, theta_stack);
        model_image_stack(*model, theta_stack,im_stack);
        for(int n=0;n<this->num_tests;n++){
            auto im=model_image(*model,model->make_stencil(theta_stack.col(n)));
            EXPECT_PRED1(&::vec_all_nonneg<typename TypeParam::ImageT>, im_stack.slice(n))<<
                                                        "Image has negative values";
            EXPECT_PRED2(&::vec_near<typename TypeParam::ImageT>,im_stack.slice(n),im)<<
                                                        "Stack simulated image";
        }
    }
}

TYPED_TEST(PointEmitterModelTest, SimulateImageStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        int nimages=256;
        // Test multiple images from 1 theta
        for(int n=0;n<10;n++){
            auto theta=model->sample_prior(this->rng);
            auto model_im=model_image(*model,model->make_stencil(theta));
            auto ims=model->make_image_stack(nimages);
            simulate_image_stack(*model,theta,ims);
            for(int i=1;i<nimages;i++){
                EXPECT_PRED1(&::vec_all_nonneg<typename TypeParam::ImageT>, ims.slice(i))<<
                                        "Image has negative values";
                EXPECT_PRED2(&::vec_ne<typename TypeParam::ImageT>,ims.slice(i), ims.slice(i-1))<<
                                        "successive image simulations identical";
            }
        }
        // Test multiple thetas
        auto im_stack=model->make_image_stack(nimages);
        auto theta_stack=model->make_param_stack(nimages);
        sample_prior_stack(*model, theta_stack);
        simulate_image_stack(*model,theta_stack,im_stack);
        for(int i=1;i<nimages;i++){
            EXPECT_PRED1(&::vec_all_nonneg<typename TypeParam::ImageT>,im_stack.slice(i))<<
                                    "Image has negative values";
            EXPECT_PRED2(&::vec_ne<typename TypeParam::ImageT>,im_stack.slice(i), im_stack.slice(i-1))<<
                                    "different thetas give identical image simulations ";
        }
    }
}


TYPED_TEST(PointEmitterModelTest, LLHStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto llh_stack=VecT(this->num_tests);
        auto im_stack=model->make_image_stack(this->num_tests);
        auto theta_stack=model->make_param_stack(this->num_tests);
        sample_prior_stack(*model, theta_stack);
        simulate_image_stack(*model,theta_stack,im_stack);
        log_likelihood_stack(*model,im_stack,theta_stack,llh_stack);
        for(int n=1;n<this->num_tests;n++){
            EXPECT_LE(llh_stack(n),0)<<"LLH values negative";
            EXPECT_NE(llh_stack(n),llh_stack(n-1))<<"Successive LLH values differ";
            typename TypeParam::ParamT theta=theta_stack.col(n);
            double llh_1=log_likelihood(*model,im_stack.slice(n), theta);
            EXPECT_EQ(llh_stack(n),llh_1)<<"LLH stack comp not equal to individual comp with theta";
            auto s=model->make_stencil(theta);
            double llh_2=log_likelihood(*model,im_stack.slice(n), s);
            EXPECT_EQ(llh_stack(n),llh_2)<<"LLH stack comp not equal to individual comp with stencil";
        }
    }
}

TYPED_TEST(PointEmitterModelTest, LLHStackSingleTheta)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto llh_stack=VecT(this->num_tests);
        auto im_stack=model->make_image_stack(this->num_tests);
        auto theta=model->sample_prior(this->rng);
        auto s=model->make_stencil(theta);
        simulate_image_stack(*model,theta,im_stack);
        log_likelihood_stack(*model,im_stack,theta,llh_stack);
        for(int n=1;n<this->num_tests;n++){
            EXPECT_LE(llh_stack(n),0)<<"LLH values negative";
            EXPECT_NE(llh_stack(n),llh_stack(n-1))<<"Successive LLH values differ";
            double llh_1=log_likelihood(*model,im_stack.slice(n), theta);
            EXPECT_EQ(llh_stack(n),llh_1)<<"LLH stack comp not equal to individual comp with theta";
            double llh_2=log_likelihood(*model,im_stack.slice(n), s);
            EXPECT_EQ(llh_stack(n),llh_2)<<"LLH stack comp not equal to individual comp with stencil";
        }
    }
}

TYPED_TEST(PointEmitterModelTest, LLHStackSingleImage)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto llh_stack=VecT(this->num_tests);
        auto im=model->make_image_stack(1);
        auto theta_stack=model->make_param_stack(this->num_tests);
        sample_prior_stack(*model, theta_stack);
        simulate_image_stack(*model,theta_stack.col(0),im);
        log_likelihood_stack(*model,im,theta_stack,llh_stack);
        for(int n=1;n<this->num_tests;n++){
            EXPECT_LE(llh_stack(n),0)<<"LLH values negative";
            EXPECT_NE(llh_stack(n),llh_stack(n-1))<<"Successive LLH values differ";
            typename TypeParam::ParamT theta=theta_stack.col(n);
            double llh_1=log_likelihood(*model, im.slice(0), theta);
            EXPECT_EQ(llh_stack(n),llh_1)<<"LLH stack comp not equal to individual comp with theta";
            auto s=model->make_stencil(theta);
            double llh_2=log_likelihood(*model,im.slice(0), s);
            EXPECT_EQ(llh_stack(n),llh_2)<<"LLH stack comp not equal to individual comp with stencil";
        }
    }
}


TYPED_TEST(PointEmitterModelTest, CRLBStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        typename TypeParam::ParamVecT theta_stack=model->make_param_stack(this->num_tests);
        typename TypeParam::ParamVecT crlb_stack=model->make_param_stack(this->num_tests);
        sample_prior_stack(*model, theta_stack);
        cr_lower_bound_stack(*model, theta_stack, crlb_stack);
        for(int n=1;n<this->num_tests;n++){
            EXPECT_GT(crlb_stack(n),0)<<"LLH values negative";
            EXPECT_PRED2(&::vec_ne<typename TypeParam::ParamT>, crlb_stack.col(n),crlb_stack.col(n-1))<<
                "Successive CRLB values differ";
            typename TypeParam::ParamT theta=theta_stack.col(n);
            auto crlb_1=cr_lower_bound(*model,theta);
            EXPECT_PRED2(&::vec_near<typename TypeParam::ParamT>, crlb_stack.col(n), crlb_1)<<
                "CRLB stack comp not equal to individual comp (from theta)";
            auto s=model->make_stencil(theta);
            auto crlb_2=cr_lower_bound(*model,s);
            EXPECT_PRED2(&::vec_near<typename TypeParam::ParamT>, crlb_stack.col(n), crlb_2)<<
                "CRLB stack comp not equal to individual comp (from stencil)";
        }
    }
}


TYPED_TEST(PointEmitterModelTest, EstimateStack)
{
    for(auto model_it=this->models.begin(); model_it!=this->models.end(); ++model_it) {
        auto model=*model_it;
        auto theta=model->make_param_stack(this->num_tests);
        auto theta_est=model->make_param_stack(this->num_tests);
        auto crlb=model->make_param_stack(this->num_tests);
        auto ims=model->make_image_stack(this->num_tests);
        auto llh=VecT(this->num_tests);
        sample_prior_stack(*model, theta);
        simulate_image_stack(*model,theta, ims);
        for(const std::string name: model->estimator_names) {
            auto estimator=make_estimator(*model, name);
            ASSERT_TRUE(bool(estimator))<<"Failed to find named estimator: "<<name;
            estimator->estimate_stack(ims, theta_est, crlb, llh);
            //Check estimates obey bounds
            if(name.find("CGauss")==std::string::npos){
                //CGauss violates this constraint
                for(int n=0;n<this->num_tests;n++) {
                    check_bounds<TypeParam>(*model,theta_est.col(n));
                }
            }
            if(name!="CGaussMLE") {
                //CGauss does some crazy stuff here
                //Check CRLB
                typename TypeParam::ParamVecT crlb_alt=model->make_param_stack(this->num_tests);
                cr_lower_bound_stack(*model, theta_est, crlb_alt);
                EXPECT_PRED2(&::vec_near<typename TypeParam::ParamVecT>, crlb_alt, crlb)<<
                        "CRLB matches independent computation: Estimator:"<<name;
                //Check LLH
                VecT llh_alt=VecT(this->num_tests);
                log_likelihood_stack(*model,ims,theta_est,llh_alt);
                EXPECT_PRED2(&::vec_near<VecT>, llh_alt, llh)<<
                        "LLH matches independent computation: Estimator:"<<name;
            }
        }
    }
}


        
}  // namespace

int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
