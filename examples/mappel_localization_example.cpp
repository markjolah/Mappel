/*
 * Example MAPPEL C++ program.
 * Samples Nsample parameter values from the prior, simulates Nsample images with Poisson noise, and localizes them.
 *
 */

#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <thread>
#include "omp.h"

#include "Mappel/Gauss2DMAP.h"
// useage:
//  <progname> [NSample] [theta_params ... ]
int main(int argc, char** argv){
    omp_set_num_threads(std::thread::hardware_concurrency());
    int Nsample = 1000;
    if(argc>1) Nsample = atoi(argv[1]);

    //Create model object
    arma::uvec size = {8, 8}; // [sizeX, sizeY]
    arma::vec psf_sigma ={0.95, 0.95}; // [psf_sigmaX, psf_sigmaY]
    mappel::Gauss2DMAP model(size,psf_sigma);
    int N = model.get_num_params();
    std::string method = "QuasiNewton";

    //Sample theta values
    auto theta = model.sample_prior();
    for(int n=0; n<N; n++) if(argc>n+2) theta(n) = std::atof(argv[n+2]); //Override with command line args
    std::cout<<"Model: "<<model.name<<" size: ["<<size(0)<<","<<size(1)<<"] psf_sigma:"<<psf_sigma(0)<<","<<psf_sigma(1)<<"]\n";
    std::cout<<"Estimating "<<Nsample<<" samples.\n";
    std::cout<<"Theta: "<<theta.t()<<std::endl;
    std::cout<<"Method: "<<method<<std::endl;

    //Simulate a stack of images
    auto ims = model.make_image_stack(Nsample);
    mappel::methods::simulate_image_stack(model,theta,ims);
    mappel::estimator::MLEDataStack mle;

    mappel::StatsT stats;
    mappel::methods::estimate_max_stack(model,ims,method,mle,stats);

    double confidence = 0.95; //Confidence interval size
    auto theta_lb = model.make_param();
    auto theta_ub = model.make_param();
    mappel::methods::error_bounds_expected(model, theta, confidence, theta_lb, theta_ub);
    arma::vec lb_outlier_p =  arma::mean(arma::conv_to<arma::mat>::from(mle.theta < arma::repmat(theta_lb,1,Nsample)),1);
    arma::vec ub_outlier_p =  arma::mean(arma::conv_to<arma::mat>::from(mle.theta > arma::repmat(theta_ub,1,Nsample)),1);

    auto theta_mean = arma::mean(mle.theta,1);
    auto theta_stddev = sqrt(arma::var(mle.theta,0,1));

    std::cout<<"[[[ Stats ]]]\n";
    for(auto& s: stats) std::cout<<s.first<<": "<<s.second<<std::endl;
    std::cout<<"[[[ Accuracy ]]]\n";
    std::cout<<"Theta:     "<<theta.t();
    std::cout<<"ThetaLB:   "<<theta_lb.t();
    std::cout<<"ThetaUB:   "<<theta_ub.t();
    std::cout<<"ThetaMean: "<<theta_mean.t();
    std::cout<<"ThetaSTD:  "<<theta_stddev.t();
    std::cout<<"LB outlier prob:"<<lb_outlier_p.t();
    std::cout<<"UB outlier prob:"<<ub_outlier_p.t();

    return 0;
}
