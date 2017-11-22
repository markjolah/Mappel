#include <cstdlib>
#include <ctime>
#include <iostream>

#include "display.h"
#include "evaluate.h"
#include "Gauss1DMLE.h"
//#include "Gauss2DMLE.h"
// #include "Gauss2DsMAP.h"
// #include "Gauss2DsMLE.h"
// #include "Blink2DsMAP.h"
// #include "GaussHSMAP.h"
// #include "GaussHSsMAP.h"
// #include "BlinkHSsMAP.h"


using namespace std;
using namespace mappel;

#define DEFAULT_PSF_SIGMA 1
#define DEFAULT_BG 0.1
#define DEFAULT_I  250
#define DEFAULT_NUM_TRIALS 100
#define DEFAULT_NUM_POINTS 50
#define DEFAULT_NUM_IMAGES 256


std::vector<std::string> model_names= {"Gauss1DMLE"};
std::vector<std::string> estimator_names= {"TrustRegionMaximizer", "HeuristicEstimator", "CGaussHeuristicEstimator", "CGaussMLE", "SimulatedAnnealingMaximizer"};



template<class Model, typename= typename std::enable_if<std::is_base_of<Gauss1DModel,Model>::value>::type >
typename Model::ParamT read_theta(Model &model, int argc, const char *argv[])
{
    // args: x I bg
    int n=3;
    double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
    int I        = argc>=n-- ? atoi(argv[n])   : -1;
    double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    auto theta = model.sample_prior();
    if(x>=0) theta(0)=x;
    if(I>=0) theta(1)=I;
    if(bg>=0) theta(2)=bg;
    return theta;
}


// template<class Model, typename=std::enable_if<std::is_base_of<Gauss2DModel>::value>::type >
// typename Model::ParamT read_theta(Model &model, int argc, const char *argv[])
// {
//     // args: x y I bg sigma
//     int n=5;
//     double sigma = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     int I        = argc>=n-- ? atoi(argv[n])   : -1;
//     double y     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     auto theta=model.sample_prior();
//     if(x>=0) theta(0)=x;
//     if(y>=0) theta(1)=y;
//     if(I>=0) theta(2)=I;
//     if(bg>=0) theta(3)=bg;
//     if(model.get_num_params()>4 && sigma>=0) theta(4)=sigma;
//     return theta;
// }

template<class Model>
typename Model::ParamT read_HS_theta(Model &model, int argc, const char *argv[])
{
    // args: x y L I bg sigma sigmaL
    int n=7;
    double sigmaL= argc>=n-- ? strtod(argv[n],NULL) : -1;
    double sigma = argc>=n-- ? strtod(argv[n],NULL) : -1;
    double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
    int I        = argc>=n-- ? atoi(argv[n])   : -1;
    double L     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    double y     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    auto theta=model.sample_prior();
    if(x>=0) theta(0)=x;
    if(y>=0) theta(1)=y;
    if(L>=0) theta(2)=L;
    if(I>=0) theta(3)=I;
    if(bg>=0) theta(4)=bg;
    if(sigma>=0) theta(5)=sigma;
    if(sigmaL>=0) theta(6)=sigmaL;
//     if(model.num_params>5 && sigma>=0) theta(4)=sigma;
    return theta;
}
/*
template<class Model>
void evaluate_estimator_wrapper(const char *estimator_name,int argc, const char *argv[] )
{
    int n=3;
    int ntrials                = argc>=n-- ? atoi(argv[n]) : 256;
    int npoints                = argc>=n-- ? atoi(argv[n]) : 64;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    cout<<"Size: "<<size<<endl;
    cout<<"Ntrials: "<<ntrials<<endl;
    cout<<"Npoints: "<<npoints<<endl;
    argc-=3; argv+=3;
    
    typename Model::ImageSizeVecT sizes={size,size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);

    auto estimator=make_estimator(model, estimator_name);
    if(!estimator) return;
    auto eff=model.make_param_vec(npoints);
    auto pval=VecT(npoints);
    evaluate_estimator(*estimator, ntrials, npoints, eff, pval);
}

template<class Model>
void evaluate_HS_estimator_wrapper(const char *estimator_name,int argc, const char *argv[] )
{
    int n=3;
    int ntrials                = argc>=n-- ? atoi(argv[n]) : 256;
    int npoints                = argc>=n-- ? atoi(argv[n]) : 64;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    cout<<"Size: "<<size<<endl;
    cout<<"Ntrials: "<<ntrials<<endl;
    cout<<"Npoints: "<<npoints<<endl;
    argc-=3; argv+=3;
    
    typename Model::ImageSizeVecT sizes={size,size,size};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);
    auto estimator = make_estimator(model, estimator_name);
    if(!estimator) return;
    auto eff=model.make_param_vec(npoints);
    auto pval=VecT(npoints);
    evaluate_estimator(*estimator, ntrials, npoints, eff, pval);
}*/

/*
void test_evaluate_estimator(int argc, const char *argv[])
{
    //  input:  modelname estimatorname size npoints ntrials
    int n=2;
    const char *estimator_name = argc>=n-- ? argv[n]       : "Newton";
    const char *model_name     = argc>=n-- ? argv[n]       : "Gauss2DMLE";
    cout<<"Model Name: "<<model_name<<endl;
    cout<<"Estimator Name: "<<estimator_name<<endl;
    argc-=2; argv+=2;
    if(istarts_with(model_name,"Gauss1DMLE")) {
        evaluate_estimator_wrapper<Gauss1DMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         evaluate_estimator_wrapper<Gauss2DMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         evaluate_estimator_wrapper<Gauss2DsMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         evaluate_estimator_wrapper<Gauss2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         evaluate_estimator_wrapper<Blink2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSMAP")) {
//         evaluate_HS_estimator_wrapper<GaussHSMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         evaluate_HS_estimator_wrapper<GaussHSsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         evaluate_HS_estimator_wrapper<BlinkHSsMAP>(estimator_name, argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}*/



template<class Model>
void point_evaluate_estimator_wrapper(const char *estimator_name, int argc, const char *argv[])
{
    int n=2;
    int count = argc>=n-- ? atoi(argv[n]) : 100;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    cout<<"Size: "<<size<<endl;
    cout<<"Count: "<<count<<endl;
    argc-=2; argv+=2;
    typename Model::ImageSizeVecT sizes={size,size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);

    auto theta=read_theta(model, argc, argv);
    auto eff=model.make_param();
    double pval;
    auto estimator=make_estimator(model, estimator_name);
    if(!estimator) return;
    point_evaluate_estimator(*estimator, count, theta, eff, pval);
}

template<class Model>
void point_evaluate_HS_estimator_wrapper(const char *estimator_name, int argc, const char *argv[])
{
    int n=2;
    int count = argc>=n-- ? atoi(argv[n]) : 100;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    cout<<"Size: "<<size<<endl;
    cout<<"Count: "<<count<<endl;
    argc-=2; argv+=2;

    typename Model::ImageSizeVecT sizes={size,size,size};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);
    auto theta=read_HS_theta(model, argc, argv);
    auto eff=model.make_param();
    double pval;
    auto estimator=make_estimator(model, estimator_name);
    if(!estimator) return;
    point_evaluate_estimator(*estimator, count, theta, eff, pval);
}



void test_point_evaluate_estimator(int argc, const char *argv[])
{
    //   input:  modelname estimatorname size #images x y I bg size psf_sigma
    int n=2;
    const char *estimator_name = argc>=n-- ? argv[n] : "Newton";
    const char *model_name     = argc>=n-- ? argv[n] : "Gauss2DMLE";
    cout<<"Model Name: "<<model_name<<endl;
    cout<<"Estimator Name: "<<estimator_name<<endl;
    argc-=2; argv+=2;
    if(istarts_with(model_name,"Gauss1DMLE")) {
        point_evaluate_estimator_wrapper<Gauss1DMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         point_evaluate_estimator_wrapper<Gauss2DMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         point_evaluate_estimator_wrapper<Gauss2DsMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         point_evaluate_estimator_wrapper<Gauss2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         point_evaluate_estimator_wrapper<Blink2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSMAP")) {
//         point_evaluate_HS_estimator_wrapper<GaussHSMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         point_evaluate_HS_estimator_wrapper<GaussHSsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         point_evaluate_HS_estimator_wrapper<BlinkHSsMAP>(estimator_name, argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}

template<class Model> 
void compare_estimators_wrapper(int argc, const char *argv[])
{
    int n=2;
    int count = argc>=n-- ? atoi(argv[n]) : 100;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=2; argv+=2;
    cout<<"Count: "<<count<<endl;
    cout<<"Size: "<<size<<endl;
    typename Model::ImageSizeVecT sizes={size,size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);

    auto theta=read_theta(model, argc, argv);
    compare_estimators(model, theta,  count);
}

template<class Model>
void compare_HS_estimators_wrapper(int argc, const char *argv[])
{
    int n=2;
    int count = argc>=n-- ? atoi(argv[n]) : 100;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=2; argv+=2;
    typename Model::ImageSizeVecT sizes={size,size,size};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Count: "<<count<<endl;
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);
    auto theta=read_HS_theta(model, argc, argv);
    compare_estimators(model, theta, count);
}


void test_compare_estimators(int argc, const char *argv[])
{
    // args: modelname size #images x y I bg sigma
    int n=1;
    const char *model_name = argc>=n-- ? argv[n] : "Gauss2DMLE";
    cout<<"Model Name: "<<model_name<<endl;
    argc-=1; argv+=1;
    if(istarts_with(model_name,"Gauss1DMLE")) {
        compare_estimators_wrapper<Gauss1DMLE>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         compare_estimators_wrapper<Gauss2DMAP>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         compare_estimators_wrapper<Gauss2DsMLE>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         compare_estimators_wrapper<Gauss2DsMAP>(argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         compare_estimators_wrapper<Blink2DsMAP>(argc, argv);
//      } else if(istarts_with(model_name,"GaussHSMAP")) {
//         compare_HS_estimators_wrapper<GaussHSMAP>(argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         compare_HS_estimators_wrapper<GaussHSsMAP>(argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         compare_HS_estimators_wrapper<BlinkHSsMAP>(argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}


template<class Model>
void compare_estimators_single_wrapper(int argc, const char *argv[])
{
    int n=1;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=1; argv+=1;
    cout<<"Size: "<<size<<endl;
    typename Model::ImageSizeVecT sizes={size,2+size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);

    auto theta=read_theta(model, argc, argv);
    compare_estimators_single(model, theta);
}

template<class Model>
void compare_HS_estimators_single_wrapper(int argc, const char *argv[])
{
    int n=1;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=1; argv+=1;
    typename Model::ImageSizeVecT sizes={size,2+size,4+size};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);auto theta=read_HS_theta(model, argc, argv);
    compare_estimators_single(model, theta);
}

void test_image_compare_estimators(int argc, const char *argv[])
{
    // args: modelname size x y I bg sigma
    int n=1;
    const char *model_name     = argc>=n-- ? argv[n]       : "Gauss2DMLE";
    cout<<"Model name: "<<model_name<<endl;
    argc-=1; argv+=1;
    if(istarts_with(model_name,"Gauss1DMLE")) {
        compare_estimators_single_wrapper<Gauss1DMLE>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         compare_estimators_single_wrapper<Gauss2DMAP>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         compare_estimators_single_wrapper<Gauss2DsMLE>(argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         compare_estimators_single_wrapper<Gauss2DsMAP>(argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         compare_estimators_single_wrapper<Blink2DsMAP>(argc, argv);
//     } else if(istarts_with(model_name,"GaussHSMAP")) {
//         compare_HS_estimators_single_wrapper<GaussHSMAP>(argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         compare_HS_estimators_single_wrapper<GaussHSsMAP>(argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         compare_HS_estimators_single_wrapper<BlinkHSsMAP>(argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}



template<class Model>
void test_image_template(const char *estimator_name,int argc, const char *argv[])
{
    int n=1;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=1; argv+=1;
    cout<<"Size: "<<size<<endl;
    typename Model::ImageSizeVecT sizes={size,size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);

    auto estimator=make_estimator(model, estimator_name);
    if(!estimator) return;
    auto theta=read_theta(model, argc, argv);
    evaluate_single(*estimator, theta);
}

template<class Model>
void test_HS_image_template(const char *estimator_name, int argc, const char *argv[])
{
    int n=1;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    argc-=1; argv+=1;
    typename Model::ImageSizeVecT sizes={size,size+2,size+4};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);    auto estimator=make_estimator(model, estimator_name);
    auto theta=read_HS_theta(model, argc, argv);
    evaluate_single(*estimator, theta);
}

// 
void test_image(int argc, const char *argv[])
{
    // args: modelname estimatorname #images size x y I bg sigma
    int n=2;
    const char *estimator_name = argc>=n-- ? argv[n] : "@";
    const char *model_name     = argc>=n-- ? argv[n]       : "Gauss2DMLE";
    cout<<"Model Name: "<<model_name<<endl;
    cout<<"Estimator Name: "<<estimator_name<<endl;
    argc-=2; argv+=2;
    if(strstr(estimator_name,"@")) {
        argc+=2; argv-=2;
//         const std::vector<std::string> &estimator_names= PoissonNoise2DObjective::estimator_names;
//         (icontains(model_name, "2D")) ? PoissonNoise2DObjective::estimator_names : PointEmitterHSModel::estimator_names;
        for(auto name: estimator_names) {
            const char *ename=name.c_str();
            argv[1]=ename;
            test_image(std::max(2,argc), argv);
            cout<<endl;
        }
    } else if(istarts_with(model_name,"Gauss1DMLE")) {
        test_image_template<Gauss1DMLE>(estimator_name,  argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         test_image_template<Gauss2DMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         test_image_template<Gauss2DsMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         test_image_template<Gauss2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         test_image_template<Blink2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSMAP")) {
//         test_HS_image_template<GaussHSMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         test_HS_image_template<GaussHSsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         test_HS_image_template<BlinkHSsMAP>(estimator_name, argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}


template<class Model>
void estimate_HS_stack_template(const char *estimator_name, int argc, const char *argv[])
{
    int n=2;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    int count                  = argc>=n-- ? atoi(argv[n]) : 1000;
    argc-=2; argv+=2;
    typename Model::ImageSizeVecT sizes={size,size,size};
    VecT sigma={0.8,1.4,1.2};
    cout<<"Sizes: "<<sizes[0]<<", "<<sizes[1]<<", "<<sizes[2]<<endl;
    cout<<"PSFsigma: "<<sigma[0]<<", "<<sigma[1]<<", "<<sigma[2]<<endl;
    Model model(sizes,sigma);
    if (istarts_with(estimator_name,"Post")){
        estimate_stack_posterior(model, count);
    } else {
        auto estimator=make_estimator(model, estimator_name);
        if(!estimator) return;
        estimate_stack(*estimator, count);
    }
}


template<class Model>
void estimate_stack_template(const char *estimator_name, int argc, const char *argv[])
{
    int n=2;
    typename Model::ImageSizeT size = argc>=n-- ? atoi(argv[n]) : 8;
    int count                  = argc>=n-- ? atoi(argv[n]) : 1000;
    argc-=2; argv+=2;
    cout<<"Size: "<<size<<endl;
    cout<<"Count: "<<count<<endl;
    typename Model::ImageSizeVecT sizes={size,size};
    VecT psf_sigma={1.0,1.0};
    Model model(sizes,psf_sigma);
    if (istarts_with(estimator_name,"Post")){
        estimate_stack_posterior(model, count);
    } else {
        auto estimator=make_estimator(model, estimator_name);
        if(!estimator) return;
        estimate_stack(*estimator, count);
    }
}

void test_speed(int argc, const char *argv[])
{
    // args: modelname estimatorname #images size
    int n=2;
    const char *estimator_name = argc>=n-- ? argv[n]       : "@";
    const char *model_name     = argc>=n-- ? argv[n]       : "Gauss2DMLE";
    argc-=2; argv+=2;
    cout<<"Model Name: "<<model_name<<endl;
    cout<<"Estimator Name: "<<estimator_name<<endl;
    if(strstr(estimator_name,"@")) {
        argc+=2; argv-=2;
//         const std::vector<std::string> &estimator_names=PoissonNoise2DObjective::estimator_names;
//             (icontains(model_name, "2D")) ? PointEmitter2DModel::estimator_names : PointEmitterHSModel::estimator_names;
        for(auto name: estimator_names) {
            const char *ename=name.c_str();
            argv[1]=ename;
            test_speed(std::max(2,argc), argv);
            cout<<endl;
        }
    } else if(istarts_with(model_name,"Gauss1DMLE")) {
        estimate_stack_template<Gauss1DMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DMAP")) {
//         estimate_stack_template<Gauss2DMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMLE")) {
//         estimate_stack_template<Gauss2DsMLE>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Gauss2DsMAP")) {
//         estimate_stack_template<Gauss2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"Blink2DsMAP")) {
//         estimate_stack_template<Blink2DsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSMAP")) {
//         estimate_HS_stack_template<GaussHSMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"GaussHSsMAP")) {
//         estimate_HS_stack_template<GaussHSsMAP>(estimator_name, argc, argv);
//     } else if(istarts_with(model_name,"BlinkHSsMAP")) {
//         estimate_HS_stack_template<BlinkHSsMAP>(estimator_name, argc, argv);
    } else {
        cout<<"Unknown model: "<<model_name<<endl;
    }
}



int main(int argc, const char *argv[]){
    const char *prog_name=argv[0];
    argc--; argv++;
    const char *model_name= argc ? argv[0] : "@";
//     cout<<"prog_name: "<< prog_name<<endl;
//     cout<<"model_name: "<<model_name<<endl;
    if(strstr(model_name,"@")) {
        for(auto name: model_names) {
            const char *cname=name.c_str();
            argv[0]=cname;
            main(std::max(2,argc+1), argv-1);
            cout<<endl;
        }
    } else {
        try{
            if(strstr(prog_name,"test_point_evaluate_estimator")) {
                test_point_evaluate_estimator(argc, argv);
            } else if (strstr(prog_name,"test_image_compare_estimators")) {
                test_image_compare_estimators(argc, argv);
            } else if (strstr(prog_name,"test_compare_estimators")) {
                test_compare_estimators(argc, argv);
            } else if (strstr(prog_name,"test_image")) {
                test_image(argc, argv);
            } else if (strstr(prog_name,"test_speed")) {
                test_speed(argc, argv);
            } else {
                cout<<"Unknown test: "<<prog_name<<endl;
            }
        } catch (MappelError &e) {
            std::cout<<"Caught MappelError: Condition:"<<e.condition()
                <<" What:"<<e.what()<<"\n"<<"Backtrace:\n"<<e.backtrace()<<"\n"; 
            return -1;
        } catch (std::exception &e) {
            std::cout<<"Caught Unhandled std::exception What:"<<e.what()<<"\n";
            return -2;
        } catch (...) {
            std::cout<<"Caught Mystery Excecption. \n";
            return -3;
        }
    }
    return 0;
}
