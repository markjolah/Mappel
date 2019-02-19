#include "Mappel/Gauss1DMAP.h"
#include <iostream>
using namespace mappel;
using namespace std;


int main(int argc, char** argv){
    int Nsample = 10000;
    if(argc>1) Nsample = atoi(argv[1]);
    const int size = 8;
    const double psf_sigma = 1;
    Gauss1DMAP model(size,psf_sigma);
    auto thetas = model.make_param_stack(Nsample);
    methods::sample_prior_stack(model,thetas);
    auto ims = model.make_image_stack(Nsample);
    methods::simulate_image_stack(model,thetas,ims);
    std::string method = "NewtonDiagonal";
    auto theta_init = model.make_param_stack(Nsample);
    auto theta_max = model.make_param_stack(Nsample);
    theta_init.zeros();
    VecT llh(Nsample);
    auto obsI = model.make_param_mat_stack(Nsample);
    StatsT stats;
    std::cout<<"Running "<<Nsample<<" samples."<<std::endl;
    methods::estimate_max_stack(model,ims,method,theta_init,theta_max,llh,obsI,stats);
    std::cout<<"[[[ Stats ]]]\n";
    for(auto& s: stats) std::cout<<s.first<<": "<<s.second<<std::endl;
    
    return 0;
}
