/** @file estimator_impl.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 01-15-2014
 * @brief 
 */
#ifndef _MAPPEL_ESTIMATOR_IMPL_H
#define _MAPPEL_ESTIMATOR_IMPL_H

#include <cmath>
#include <armadillo>

#include "estimator.h"
#include "rng.h"
#include "numerical.h"

#ifdef WIN32
    using namespace boost::chrono;
#else
    using namespace std::chrono;
#endif
/**
 *
 * All models will call for maximization through this virtual function.
 * All non-GPU based maximizers will use this version which spawns threads
 * using a non-virual entry point member function Maximizer::thread_entry.
 * GPU-based maximizers will want to do something custom, so they will declare
 * their own virtual maximize_stack.
 *
 * It is also because of the GPU-based mamixmizers that we are putting initilization,
 * and CRLB/LLH calculations in here even though the Model knows how to do them.
 *
 * We expect that those methods will need to also be paralellized and the GPU will need
 * custom code, and the threaded CPU versions will want to also compute those in parallel,
 * so in order to have a consitent call interface to the Maximizer classes,
 * we put the CRLB/LLH and initialization work within the the maximize_stack method.
 *
 *
 */

namespace mappel {

template<class Model>
Model& Estimator<Model>::get_model()
{ return model; }

template<class Model>
void Estimator<Model>::set_model(Model &new_model)
{ 
    model = new_model;
}

/* Option 1: Single Image - Estimator returns a stencil at the optimum point.
 * Variants with and without a theta_init or a rllh output parameter
 */
template<class Model>
StencilT<Model>
Estimator<Model>::estimate_max(const ModelDataT<Model> &im)
{
    auto theta_init = model.make_param();
    theta_init.zeros();
    return estimate_max(im, theta_init);
}

template<class Model>
StencilT<Model>
Estimator<Model>::estimate_max(const ModelDataT<Model> &im, double &rllh)
{
    auto theta_init = model.make_param();
    theta_init.zeros();
    return estimate_max(im, theta_init,rllh);
}

template<class Model>
StencilT<Model>
Estimator<Model>::estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init)
{
    double rllh;
    return estimate_max(im,theta_init,rllh);
}

template<class Model>
StencilT<Model>
Estimator<Model>::estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh)
{
    auto start_walltime = ClockT::now();
    StencilT<Model> est = compute_estimate(im, theta_init, rllh);
    record_walltime(start_walltime, 1);
    return est;
}

/* Option 2: Single Image - Estimator returns theta, crlb and llh  at the optimum point.
 * Variants with and without a theta_init.
 */
template<class Model>
void Estimator<Model>::estimate_max(const ModelDataT<Model> &im,
                                ParamT<Model> &theta, double &rllh, MatT &obsI)
{
    auto theta_init = model.make_param();
    theta_init.zeros();
    estimate_max(im, theta, theta_init, rllh, obsI);
}

template<class Model>
void Estimator<Model>::estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
                                    ParamT<Model> &theta, double &rllh, MatT &obsI)
{
    auto start_walltime = ClockT::now();
    compute_estimate(im, theta_init, theta, rllh, obsI);
    record_walltime(start_walltime, 1);
}

/* Option 3: Single Image Debug mode - Estimator returns theta, rllh and obsI  at the optimum point.
 */
template<class Model>
void Estimator<Model>::estimate_max_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                          ParamT<Model> &theta_est, double &rllh, MatT &obsI, MatT &sequence, VecT &sequence_rllh)
{
    auto start_walltime = ClockT::now();
    auto est_max = compute_estimate_debug(im, theta_init, sequence, sequence_rllh);
    theta_est = est_max.theta;
    rllh = methods::objective::rllh(model,im,est_max);
    obsI = methods::observed_information(model,im,est_max);
    record_walltime(start_walltime, 1);
}

template<class Model>
void Estimator<Model>::estimate_max_stack(const ModelDataStackT<Model> &im,
                                         ParamVecT<Model> &theta_max_stack, VecT &rllh_stack, CubeT &obsI_stack)
{
    ParamVecT<Model> theta_init = model.make_param();
    theta_init.zeros();
    estimate_max_stack(im, theta_init, theta_max_stack, rllh_stack, obsI_stack);
}

/** @brief Default base class implementation computes rllh and obsI seperately from stencil 
 * This should be overridden by Estimator subclasses that already have access to this information
 */
template<class Model>
void Estimator<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
                                        ParamT<Model> &theta_max, double &rllh, MatT &obsI)
{
    auto est_max = compute_estimate(im, theta_init, rllh);
    obsI = methods::observed_information(model, im, est_max);
    theta_max = est_max.theta;
}

template<class Model>
StatsT Estimator<Model>::get_stats()
{
    StatsT stats;
    stats["num_estimations"] = num_estimations;
    stats["total_walltime"] = total_walltime;
    return stats;
}

template<class Model>
void Estimator<Model>::clear_stats()
{
    num_estimations = 0;
    total_walltime = 0.;
}

template<class Model>
std::ostream& operator<<(std::ostream &out, Estimator<Model> &estimator)
{
    auto stats=estimator.get_stats();
    out<<"[Estimator: "<<estimator.name()<<"<"<<estimator.model.name<<">]\n";
    for(auto& stat: stats) out<<"\t"<<stat.first<<"="<<stat.second<<"\n";
    return out;
}


/**
 *
 * Estimators that produce a sequence of results (e.g. IterativeEstimators) can override this
 * dummy debug implementation.
 */
template<class Model>
inline
StencilT<Model>
Estimator<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                         ParamVecT<Model> &sequence, VecT &sequence_rllh)
{
    sequence = model.make_param_stack(1);
    sequence_rllh.set_size(1);
    auto est = compute_estimate(im,theta_init,sequence_rllh(0));
    sequence.col(0) = est.theta;
    return est;
}

template<class Model>
void Estimator<Model>::record_walltime(ClockT::time_point start_walltime, int nimages)
{
    double walltime = duration_cast<duration<double>>(ClockT::now() - start_walltime).count();
    total_walltime += walltime;
    num_estimations += nimages;
}

/* Threaded Estimator */ 

template<class Model>
ThreadedEstimator<Model>::ThreadedEstimator(Model &model)
    : Estimator<Model>(model),
      max_threads(omp_get_max_threads()),
      num_threads(1)
{ }



template<class Model>
void ThreadedEstimator<Model>::estimate_max_stack(const ModelDataStackT<Model> &im_stack, const ParamVecT<Model> &theta_init_stack,
                                              ParamVecT<Model> &theta_est_stack, VecT &rllh_stack, CubeT &obsI_stack)
{
    auto start_walltime=ClockT::now();
    IdxT Nimages = model.get_size_image_stack(im_stack);
    #pragma omp parallel
    {
        if(omp_get_thread_num()==0) num_threads = omp_get_num_threads();
        auto theta_est = model.make_param();
        auto obsI = model.make_param_mat();
        #pragma omp for
        for(IdxT n=0; n<Nimages; n++) {
            this->compute_estimate(model.get_image_from_stack(im_stack,n), theta_init_stack.col(n), theta_est, rllh_stack(n), obsI);
            theta_est_stack.col(n) = theta_est;
            obsI_stack.slice(n) = obsI;
        }
    } 
    this->record_walltime(start_walltime, Nimages);
}

template<class Model>
StatsT ThreadedEstimator<Model>::get_stats()
{
    auto stats = Estimator<Model>::get_stats();
    stats["num_threads"] = num_threads;
    stats["total_thread_time"] = num_threads*this->total_walltime;
    stats["mean_estimation_time"] =  num_threads*this->total_walltime / this->num_estimations;
    return stats;
}

template<class Model>
StatsT ThreadedEstimator<Model>::get_debug_stats()
{
    return ThreadedEstimator<Model>::get_stats();
}

template<class Model>
void ThreadedEstimator<Model>::clear_stats()
{
    Estimator<Model>::clear_stats();
    num_threads=1;
}

/* HeuristicEstimator */
template<class Model>
StencilT<Model> 
HeuristicEstimator<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh)
{
    auto est = Estimator<Model>::model.initial_theta_estimate(im,theta_init);
    rllh = methods::objective::rllh(this->model, im,est);
    return est;
}

/* CGaussHeuristicEstimator */
template<class Model>
StencilT<Model> 
CGaussHeuristicEstimator<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh) 
{
    auto est = cgauss_heuristic_compute_estimate(model,im,theta_init);
    rllh = methods::objective::rllh(this->model, im,est);
    return est;
}

/* CGaussMLE */
template<class Model>
StatsT CGaussMLE<Model>::get_stats()
{
    auto stats = ThreadedEstimator<Model>::get_stats();
    stats["mean_iterations"] = max_iterations;
    stats["mean_backtracks"] = 0;
    stats["mean_fun_evals"] = 0;
    stats["mean_der_evals"] = max_iterations;
    return stats;
}

template<class Model>
StatsT CGaussMLE<Model>::get_debug_stats()
{
    StatsT stats =  CGaussMLE<Model>::get_stats();
    stats["debugIterative"] = 0; //No backtracks so no need to notate them
    return stats;
}


template<class Model>
StencilT<Model>
CGaussMLE<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh) 
{
    auto est=cgauss_compute_estimate(model,im,theta_init,max_iterations);
    rllh = methods::objective::rllh(model,im,est);
    return est;
}

template<class Model>
StencilT<Model>
CGaussMLE<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                         ParamVecT<Model> &sequence, VecT &sequence_rllh) 
{
    auto est = cgauss_compute_estimate_debug(model,im,theta_init,max_iterations,sequence);
    //Compute rrlh (which are not evaluated by CGauss) in parallel because debug is only ever called single threaded
    methods::objective::rllh_stack(model, im, sequence, sequence_rllh);
    return est;
}

/* Iterative Maximizer */

template<class Model>
IterativeMaximizer<Model>::IterativeMaximizer(Model &model, int max_iterations)
    : ThreadedEstimator<Model>(model), 
      max_iterations(max_iterations),
      exit_counts(NumExitCodes,arma::fill::zeros)
{}

template<class Model>
IterativeMaximizer<Model>::MaximizerData::MaximizerData(const Model &model, const ModelDataT<Model> &im,
                                                const StencilT<Model> &s, bool save_seq, int max_seq_len)
    : im(im), 
      grad(model.make_param()), 
      lbound(model.get_lbound()),
      ubound(model.get_ubound()),
      rllh(methods::objective::rllh(model,im,s)), 
      save_seq(save_seq), 
      s0(s), 
      current_stencil(true),
      max_seq_len(max_seq_len)
{
    if (save_seq){
        theta_seq = model.make_param_stack(max_seq_len);
        backtrack_idxs.set_size(max_seq_len);
        backtrack_idxs.zeros();
        seq_rllh.set_size(max_seq_len);
        seq_rllh.zeros();
        record_iteration(); //record this initial point
    }
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::record_iteration(const ParamT<Model> &accpeted_theta)
{
    nIterations++;
    if(save_seq) {
        if(seq_len>=max_seq_len) throw LogicalError("Exceeded MaximizerData sequence limit");
        theta_seq.col(seq_len) = accpeted_theta;
        seq_rllh(seq_len) = rllh;
        seq_len++;
    }
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::record_backtrack(const ParamT<Model> &rejected_theta, double rejected_rllh)
{
    nBacktracks++;
    if(save_seq) {
        if(seq_len>=max_seq_len) throw LogicalError("Exceeded MaximizerData sequence limit");
        theta_seq.col(seq_len) = rejected_theta;
        backtrack_idxs(seq_len) = 1;
        seq_rllh(seq_len) = rejected_rllh;
        seq_len++;
    }
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::record_exit(ExitCode code)
{
    exit_code = code;
}

template<class Model>
StatsT IterativeMaximizer<Model>::get_stats()
{
    auto stats = ThreadedEstimator<Model>::get_stats();
    mtx.lock();
    double N = static_cast<double>(this->num_estimations);
    stats["total_iterations"] = total_iterations;
    stats["total_backtracks"] = total_backtracks;
    stats["total_fun_evals"] = total_fun_evals;
    stats["total_der_evals"] = total_der_evals;
    stats["mean_iterations"] = total_iterations/N;
    stats["mean_backtracks"] = total_backtracks/N;
    stats["mean_fun_evals"] = total_fun_evals/N;
    stats["mean_der_evals"] = total_der_evals/N;
    stats["const_fval_epsilon"] = epsilon;
    stats["const_step_size_delta"] = delta;
    stats["const_max_iterations"] = max_iterations;
    stats["const_max_backtracks"] = max_backtracks;
    stats["num_exit_error"] = exit_counts(static_cast<IdxT>(ExitCode::Error));
    stats["num_exit_max_iter"] = exit_counts(static_cast<IdxT>(ExitCode::MaxIter));
    stats["num_exit_max_backtracks"] = exit_counts(static_cast<IdxT>(ExitCode::MaxBacktracks));
    stats["num_exit_function_value_change"] = exit_counts(static_cast<IdxT>(ExitCode::FunctionChange));
    stats["num_exit_step_size_ratio"] = exit_counts(static_cast<IdxT>(ExitCode::StepSize));
    stats["num_exit_grad_ratio"] = exit_counts(static_cast<IdxT>(ExitCode::GradRatio));
    stats["num_exit_trust_region_radius"] = exit_counts(static_cast<IdxT>(ExitCode::TrustRegionRadius));
    mtx.unlock();
    return stats;
}

template<class Model>
StatsT IterativeMaximizer<Model>::get_debug_stats()
{
    auto stats =  IterativeMaximizer<Model>::get_stats();
    
    stats["debugIterative"]=1;
    auto backtrack_idxs = last_backtrack_idxs;
    for(unsigned n=0; n<backtrack_idxs.n_elem; n++) {
        std::ostringstream out;
        out<<"backtrack_idxs."<<n+1;
        stats[out.str()] = backtrack_idxs(n);
    }
    return stats;
}


template<class Model>
void IterativeMaximizer<Model>::clear_stats()
{
    ThreadedEstimator<Model>::clear_stats();
    mtx.lock();
    total_iterations = 0;
    total_backtracks = 0;
    total_fun_evals = 0;
    total_der_evals = 0;
    exit_counts.zeros();
    mtx.unlock();
}

template<class Model>
void IterativeMaximizer<Model>::record_run_statistics(const MaximizerData &data)
{
    mtx.lock();
    total_iterations += data.nIterations;
    total_backtracks += data.nBacktracks;
    total_fun_evals += data.nIterations + data.nBacktracks;
    total_der_evals += data.nIterations;
    exit_counts(static_cast<IdxT>(data.exit_code))++;  // Record exit code
    if(data.save_seq) last_backtrack_idxs = data.get_backtrack_idxs(); //Store the backtracks for debugging...
    mtx.unlock();
}

template<class Model>
bool IterativeMaximizer<Model>::backtrack(MaximizerData &data)
{
    double lambda = 1.0;
    data.save_stencil();  //Save current theta in case we cannot improve
    if (!data.step.is_finite()) {
        std::ostringstream msg;
        msg<<"Step is not finite: "<<data.step.t();
        throw NumericalError(msg.str());
    }
    if(arma::dot(data.grad, data.step)<=0){
        //We are maximizing so we should be moving in direction of gradiant not away
        std::ostringstream msg;
        msg<<"Backtrack with Negative Gradient. grad:"<<data.grad.t()<<" step:"<<data.step.t()<<" <grad, step>: "<<arma::dot(data.grad, data.step);
        throw NumericalError(msg.str());
    }
    for(int n=0; n<max_backtracks; n++){
        //Reflective boundary conditions
        auto new_theta = model.bounded_theta(model.reflected_theta(data.saved_theta() + lambda*data.step));
        data.set_stencil(model.make_stencil(new_theta,false));
        double can_rllh = methods::objective::rllh(model, data.im, data.stencil()); //candidate points log-lh
        bool in_bounds = model.theta_in_bounds(new_theta);

//         std::cout<<"\n [Backtrack:"<<n<<"]\n";
//         std::cout.precision(15);
//         data.saved_theta().t().raw_print(std::cout," Current Theta:");
//         std::cout<<" Step: "<<data.step.t();
//         std::cout<<" Lambda:"<<lambda<<"\n Scaled Step:"<<(lambda*data.step).t();
//         std::cout<<" Proposed Theta: "<<new_theta.t();
//         printf(" CurrentRLLH:%.9g ProposedRLLH:%.9g Delta(prop-cur):%.9g\n",data.rllh,can_rllh,can_rllh-data.rllh);

        if(!in_bounds) {
            std::ostringstream msg;
            msg<<"New theta is not inbounds. old_theta:"<<data.saved_theta().t()<<" step:"<<data.step.t()
               <<" new_theta:"<<new_theta.t();
            throw NumericalError(msg.str());
        }
        if(!std::isfinite(can_rllh)) {
            std::ostringstream msg;
            msg<<"Candidate theta is inbounds but rllh is non-finite. new_theta:"<<new_theta.t()<<" rllh:"<<can_rllh;
            throw NumericalError(msg.str());
        }

        double old_lambda = lambda; //save old lambda
        double linear_step = lambda*arma::dot(data.grad, data.step); //The amount we would go down if linear in grad
        if (can_rllh >= data.rllh + alpha*linear_step) { //Must be a sufficient increase (convergence criteria)
            //Success - Found a new point which is slightly better than where we were before, so update rllh
            data.rllh = can_rllh; //We have not yet changed data.rllh that is still the old rllh
            data.stencil().compute_derivatives(); //Now we commit to the expense of computing derivatives
            data.record_iteration();  //Record a successful point
            return false; //Tell caller to continue optimizing
        } else {
            data.record_backtrack(can_rllh); //Record a failed (backtracked) (unaccepted) point.
            if (convergence_test(data)) {
                //We converged at original iteration point.  Unable to improve by backtracking on step direction.
                //Original step was too large to converge, after backtracking it (or fun-val) became small enough to converge
                data.restore_stencil(); //Restore back to original point
                return true; 
            } else {
                //Perform Backtrack using a quadratic approximation
                double new_lambda = -.5*lambda*linear_step/(can_rllh - data.rllh - linear_step); //The minimum of a quad approx using cllh and linear_step
                double rho = new_lambda/old_lambda;  //Relative decrease
                rho = std::min(std::max(rho,0.02),0.25);  //Limit minimum and maximum relative decrease of rho
                lambda = rho*old_lambda;
            }
        }
    }
    //Backtracking failed to converge in max_backtracks steps.
    //But we otherwise did not meet convergence criteria.
    data.restore_stencil();
    data.record_exit(ExitCode::MaxBacktracks);
    return true; //Tell caller to stop and return the original theta. Backtracking failed to improve it.
}

template<class Model>
bool IterativeMaximizer<Model>::convergence_test(MaximizerData &data)
{
    using arma::norm;
    auto ntheta = data.theta();  //new theta
    auto otheta = data.saved_theta(); //old theta
    double step_size_ratio = norm(otheta-ntheta,2)/std::max(norm(otheta,2),norm(ntheta,2));
    double function_change_ratio = norm(data.grad,2)/fabs(data.rllh);
    if(step_size_ratio <= delta){
        data.record_exit(ExitCode::StepSize);
        return true; //Tell caller to stop iterating we have converged
    } else if(function_change_ratio <= epsilon) {
        data.record_exit(ExitCode::FunctionChange);
        return true; //Tell caller to stop iterating we have converged
    } else {
        return false; //Keep iterating
    }
}


template<class Model>
StencilT<Model>
IterativeMaximizer<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh)
{
    auto theta_init_stencil = this->model.initial_theta_estimate(im, theta_init);
    if(!theta_init_stencil.derivatives_computed) 
        throw LogicalError("Stencil has no computed derivatives");
    MaximizerData data(model, im, theta_init_stencil);
    try {
        maximize(data);
    } catch (NumericalError &err) {
        data.record_exit(ExitCode::Error);
    }
    record_run_statistics(data);
    rllh = data.rllh;
    return data.stencil();
}

template<class Model>
StencilT<Model>
IterativeMaximizer<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                                  ParamVecT<Model> &sequence, VecT &sequence_rllh)
{
    auto theta_init_stencil = this->model.initial_theta_estimate(im, theta_init);
    if(!theta_init_stencil.derivatives_computed) 
        throw LogicalError("Stencil has no computed derivatives");
    MaximizerData data(model, im, theta_init_stencil, true, max_iterations*max_backtracks+1);
    try {
        maximize(data);
    } catch (NumericalError &err) {
        data.record_exit(ExitCode::Error);
        std::cout<<"Caught NumericalError: "<<err.what()<<std::endl;
    }
    sequence = data.get_theta_sequence();
    sequence_rllh = data.get_theta_sequence_rllh();
    record_run_statistics(data);
    return data.stencil();
}

/* This is called to clean up simulated annealing */
template<class Model>
void IterativeMaximizer<Model>::local_maximize(const ModelDataT<Model> &im, const StencilT<Model> &theta_init, StencilT<Model> &stencil, double &rllh)
{
    MaximizerData data(model, im, theta_init);
    maximize(data);
    stencil = data.stencil();
    rllh = data.rllh;
}

template<class Model>
void NewtonDiagonalMaximizer<Model>::maximize(MaximizerData &data)
{
    double epsilon=0;
    double delta=0.1;
    auto grad2 = model.make_param();
    for(int n=0; n<this->max_iterations; n++) { //Main optimization loop
        methods::objective::grad2(model, data.im, data.stencil(), data.grad, grad2); //compute grad and diagonal hessian
        data.step = -data.grad/grad2;
        if(arma::any(grad2>0)){

//             std::cout<<"{NewtonDiagonal ITER:"<<n<<"} --- Correcting non-positive-definite\n";
//             std::cout<<"Theta:"<<data.theta().t();
//             std::cout<<"RLLH: "<<methods::objective::rllh(model, data.im, data.stencil())<<"\n";
//             std::cout<<"Grad: "<<data.grad.t();
//             std::cout<<"Grad2:"<<grad2.t();
//             std::cout<<"Step:"<<data.step.t();
//             std::cout<<"<Step,Grad>:"<<arma::dot(data.step,data.grad)<<"\n";

            //Grad2 should be negative
            double max_val = arma::max(grad2);
            grad2 -= max_val + delta;
            data.step = -data.grad/grad2;

//             std::cout<<"max_val: "<<max_val<<"\n";
//             std::cout<<"Grad2Correct:"<<grad2.t();
//             std::cout<<"Step:"<<data.step.t();
//             std::cout<<"<Step,Grad>:"<<arma::dot(data.step,data.grad)<<"\n";

            if(arma::dot(data.step,data.grad)<=epsilon) throw NumericalError("Unable to correct grad2 in NewtonDiagonal");
        }
        if(this->backtrack(data) || this->convergence_test(data)) return;  //Converged or gave up trying
    }
    data.record_exit(IterativeMaximizer<Model>::ExitCode::MaxIter);
}


template<class Model>
void NewtonMaximizer<Model>::maximize(MaximizerData &data)
{
    auto hess = model.make_param_mat();
    auto C = model.make_param_mat();
    auto Cstep = model.make_param();
    for(int n=0; n < this->max_iterations; n++) { //Main optimization loop
        methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
        data.step = arma::solve(arma::symmatu(hess), -data.grad);
        C=-hess;
        
//         copy_Usym_mat(C);
//         std::cout<<"{Newton ITER:"<<n<<"}\n";
//         std::cout.precision(15);
//         data.theta().t().raw_print(std::cout," Theta:");
//         std::cout<<"RLLH: "<<methods::objective::rllh(model, data.im, data.stencil())<<"\n";
//         std::cout<<"Grad: "<<data.grad.t();
//         std::cout<<"GradDir: "<<arma::normalise(data.grad).t();
//         std::cout<<"Hess:\n"<<arma::symmatu(hess);
//         std::cout<<"Positive-definite:"<<is_positive_definite(C)<<"\n";
        
        modified_cholesky(C);
        Cstep = cholesky_solve(C,data.grad);//Drop - sign on grad since C=-hess
        auto Cfull = C;
        cholesky_convert_full_matrix(Cfull);
        
//         std::cout<<"C:\n"<<Cfull;
//         std::cout<<"HessStep: "<<data.step.t();
//         std::cout<<"HessStepDir: "<<arma::normalise(data.step).t();
//         std::cout<<"HessStepDotGrad: "<<(arma::dot(arma::normalise(data.step),arma::normalise(data.grad)))<<"\n";
//         std::cout<<"CholStep: "<<Cstep.t();
//         std::cout<<"CholStepDir: "<<arma::normalise(Cstep).t();
//         std::cout<<"CStepDotGrad: "<<(arma::dot(arma::normalise(Cstep),arma::normalise(data.grad)))<<"\n";
//         double nrllh = methods::objective::rllh(model,data.im,data.theta()+Cstep);
//         std::cout<<"Rllh Curr: "<<data.rllh<<" CStep:"<<nrllh<<" Delta:"<<nrllh-data.rllh<<"\n";

        data.step = Cstep;
        if(!data.step.is_finite()) throw NumericalError("Bad data_step!");
        if(arma::dot(data.grad, data.step)<=0) throw NumericalError("Not an asscent direction!");
        if(this->backtrack(data) || this->convergence_test(data))  return; //Backing up did not help.  Just quit.
    }
    data.record_exit(IterativeMaximizer<Model>::ExitCode::MaxIter);
}

template<class Model>
void QuasiNewtonMaximizer<Model>::maximize(MaximizerData &data)
{
    auto grad_old=model.make_param();
    auto H=model.make_param_mat(); //This is our approximate hessian
    for(int n=0; n < this->max_iterations; n++) { //Main optimization loop
        if(n==0) {
            auto hess=model.make_param_mat();
            methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
            if(is_positive_definite(-hess)) {
                H=arma::inv(arma::symmatu(hess));
            } else {
                H=-arma::eye(model.get_num_params(), model.get_num_params());
            }
        } else {
            //Approx H
            data.grad = methods::objective::grad(model, data.im, data.stencil());
            auto delta_grad = data.grad-grad_old;
            double rho =1./arma::dot(delta_grad, data.step);
            MatT K=rho*data.step*delta_grad.t();//This is our approximate inverse hessian
            K.diag()-=1;
            H=K*H*K.t()+rho*data.step*data.step.t();
        }
        data.step=-H*data.grad;
        if(!is_positive_definite(-H)) {
            VecT lambda_H;
            MatT Q_H;
            arma::eig_sym(lambda_H,Q_H, arma::symmatu(H)); //Compute eigendecomposition of symmertic matrix H
            std::cout<<"{QuasiNewton ITER:"<<n<<"}\n";
            std::cout<<"Theta:"<<data.theta().t();
            std::cout<<"RLLH: "<<methods::objective::rllh(model, data.im, data.stencil())<<"\n";
            std::cout<<"Grad: "<<data.grad.t();
            std::cout<<"Hinv:\n"<<inv(H);
            std::cout<<"H:\n"<<H;
            std::cout<<"Lambda(H):"<<lambda_H;
            throw NumericalError("QuasiNewton: H not positive_definite");
        }
        if(!data.step.is_finite()) throw NumericalError("QuasiNewton: step is non-finite");
//         if(arma::dot(data.grad, data.step)<=0) throw Numerical("QuasiNewton: step is not a ascent direction");

        grad_old=data.grad;
        if(this->backtrack(data) || this->convergence_test(data))  return; //Backing up did not help.  Just quit.

        data.step = data.theta()-data.saved_theta();//If we backtracked then the step may have changed
    }
    data.record_exit(IterativeMaximizer<Model>::ExitCode::MaxIter);
}


template<class Model>
const double TrustRegionMaximizer<Model>::rho_cauchy_min = 0.1;  //Coleman beta | Bellavia beta_1
template<class Model>
const double TrustRegionMaximizer<Model>::rho_obj_min = 0.25;  //Coleman mu | Bellavia beta_2
template<class Model>
const double TrustRegionMaximizer<Model>::rho_obj_opt = 0.75;  //Coleman eta | Bellavia beta_2
template<class Model>
const double TrustRegionMaximizer<Model>::delta_decrease_min = 0.125; // Coleman gamma_0 | Bellavia alpha_1
template<class Model>
const double TrustRegionMaximizer<Model>::delta_decrease = 0.25; // Coleman gamma_1 | Bellavia alpha_2
template<class Model>
const double TrustRegionMaximizer<Model>::delta_increase = 2; // Coleman gamma_2 | Bellavia alpha_3

template<class Model>
const double TrustRegionMaximizer<Model>::min_scaling = 1.0e-5;  //Minimum for Dscale(i)
template<class Model>
const double TrustRegionMaximizer<Model>::max_scaling = 1.0e5;   //Maximum for Dscale(i)
template<class Model>
const double TrustRegionMaximizer<Model>::delta_init_min = 1.0e-3; //Minimum initial trust region radius
template<class Model>
const double TrustRegionMaximizer<Model>::delta_init_max = 1.0e3;  //Maximum initial trust region radius
template<class Model>
const double TrustRegionMaximizer<Model>::boundary_stepback_min_kappa = 1.0 - 1.0e-5; //Distance to step back from the bounadary to remain in interrior

template<class Model>
void TrustRegionMaximizer<Model>::maximize(MaximizerData &data)
{
    int N = static_cast<int>(data.grad.n_elem);
    auto hess = model.make_param_mat();
    auto C = model.make_param_mat();
    auto Cstep = model.make_param();
    VecT Cvec = model.make_param();
    VecT vec = model.make_param();
    VecT Dscale(N,arma::fill::zeros);
    double delta = 1.0; //Trust-radius
    for(int n=0; n<1000; n++) { //Main optimization loop
        methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
//         std::cout<<"{TR ITER:"<<n<<"}\n";
//         std::cout.precision(15);
//         data.theta().t().raw_print(std::cout," Theta:");
//         std::cout<<"   RLLH: "<<data.rllh<<"\n";
//         std::cout<<"   Grad: "<<data.grad.t();
//         std::cout<<"   Hessian:\n"<<hess;
//         std::cout<<"   -Hess is Positive-definite:"<<is_positive_definite(-hess)<<"\n";
//         
        //Compute scaling for bounding ---  Bellavia (2004); Coleman and Li (1996)
        VecT v,Jv;
        compute_bound_scaling_vec(data.theta(), -data.grad, data.lbound, data.ubound, v, Jv);
        VecT Dbound = 1./arma::sqrt(arma::abs(v));
        //Compute Scaling for problem variables
        //This is the "adaptive" method of More (1982)
//         Dscale = compute_D_scale(Dscale, hess.diag());
//         VecT D = Dscale % Dbound;
//         D.ones();
        VecT D=  Dbound;
//         D.ones();
        VecT Dinv = 1./D;
        //Establish model using negative grad and hessian for maximization
        //Substitute -hess and -grad
        //H_hat and g_hat are a minimization TR problem without scaling now.
        // smin = min(s) : ghat^T s + .5*s^T*Hhat*s
        // Then s = Dinv * Shat
        MatT Hhat = arma::diagmat(Dinv) * -arma::symmatu(hess) * arma::diagmat(Dinv) + arma::diagmat(-data.grad % Jv);
//         MatT Hhat = arma::diagmat(Dinv) * -arma::symmatu(hess) * arma::diagmat(Dinv);
        VecT ghat = Dinv % -data.grad;
        
//         std::cout<<"   delta:"<<delta<<"\n";
//         std::cout<<"   v:"<<v.t();
//         std::cout<<"   Jv:"<<Jv.t();
//         std::cout<<"   Dscale:"<<Dscale.t();
//         std::cout<<"   Dbound:"<<Dbound.t();
//         std::cout<<"   D:"<<D.t();
//         std::cout<<"   Dinv:"<<Dinv.t();
//         std::cout<<"   ghat:"<<ghat.t();
//         std::cout<<"   Hhat:\n"<<Hhat;
//         std::cout<<"   Hhat pos def:"<<is_positive_definite(Hhat)<<"\n";
//         
        
        //Compute and maintain delta in the transformed basis
        if(n==0) delta = compute_initial_trust_radius(ghat);
//         std::cout<<"   TrustRadius: "<<delta<<"\n";
        
        VecT s_hat_opt = solve_TR_subproblem(ghat, Hhat, delta, this->epsilon);
        VecT s_hat_opt_bnd = bound_step(s_hat_opt, D, data.theta(), data.lbound, data.ubound);
        VecT s_hat_cauchy = compute_cauchy_point(ghat, Hhat, delta);
        VecT s_hat_cauchy_bnd = bound_step(s_hat_cauchy, D, data.theta(), data.lbound, data.ubound);
        
//         double quad_model_opt = quadratic_model_value(s_hat_opt, ghat, Hhat);
        double quad_model_opt_bnd = quadratic_model_value(s_hat_opt_bnd, ghat, Hhat);
//         double quad_model_cauchy = quadratic_model_value(s_hat_cauchy, ghat, Hhat);
        double quad_model_cauchy_bnd = quadratic_model_value(s_hat_cauchy_bnd, ghat, Hhat);
        double rho_cauchy = quad_model_opt_bnd / quad_model_cauchy_bnd;
//         std::cout<<"  s_hat_opt:"<<s_hat_opt.t();
//         std::cout<<"  s_hat_opt_bnd:"<<s_hat_opt_bnd.t();
//         std::cout<<"  s_hat_cauchy:"<<s_hat_cauchy.t();
//         std::cout<<"  s_hat_cauchy_bnd:"<<s_hat_cauchy_bnd.t();
//         std::cout<<"  |s_hat_opt|:"<<arma::norm(s_hat_opt)<<"\n";
//         std::cout<<"  |s_hat_opt_bnd|:"<<arma::norm(s_hat_opt_bnd)<<"\n";
//         std::cout<<"  |s_hat_cauchy|:"<<arma::norm(s_hat_cauchy)<<"\n";
//         std::cout<<"  |s_hat_cauchy_bnd|:"<<arma::norm(s_hat_cauchy_bnd)<<"\n";
//         std::cout<<"  m(s_hat_opt):"<<quad_model_opt<<"\n";
//         std::cout<<"  m(s_hat_opt_bnd):"<<quad_model_opt_bnd<<"\n";
//         std::cout<<"  m(s_hat_cauchy):"<<quad_model_cauchy<<"\n";
//         std::cout<<"  m(s_hat_cauchy_bnd):"<<quad_model_cauchy_bnd<<"\n";
//         std::cout<<"  s_opt:"<<(Dinv %s_hat_opt).t();
//         std::cout<<"  s_opt_bnd:"<<(Dinv %s_hat_opt_bnd).t();
//         std::cout<<"  s_cauchy:"<<(Dinv %s_hat_cauchy).t();
//         std::cout<<"  s_cauchy_bnd:"<<(Dinv %s_hat_cauchy_bnd).t();
//         std::cout<<"  x_opt:"<<(data.theta()+Dinv %s_hat_opt).t();
//         std::cout<<"  x_opt_bnd:"<<(data.theta()+Dinv %s_hat_opt_bnd).t();
//         std::cout<<"  x_cauchy:"<<(data.theta()+Dinv %s_hat_cauchy).t();
//         std::cout<<"  x_cauchy_bnd:"<<(data.theta()+Dinv %s_hat_cauchy_bnd).t();
//         std::cout<<"  rho_cauchy:"<<rho_cauchy<<"\n";
        
        //Commit to evaluating new point
        data.save_stencil();
        double model_improvement;
        VecT s_hat;
        if(rho_cauchy < rho_cauchy_min) { // set next point as bounded cauchy point
            s_hat = s_hat_cauchy_bnd;
            model_improvement = -quad_model_cauchy_bnd;
        } else { //set next point as bounded optimal point
            s_hat = s_hat_opt_bnd;
            model_improvement = -quad_model_opt_bnd;
        }
        data.set_stencil(model.make_stencil(data.saved_theta() + Dinv % s_hat));
        double old_rllh = data.rllh;
        double can_rllh = methods::objective::rllh(model, data.im, data.stencil());
        double obj_improvement = can_rllh - old_rllh; // the "improvement" in the model.  should be positive
        double rho_obj =  obj_improvement / model_improvement;
        bool success;
        if(rho_obj < rho_obj_min) { //Backtrack Not acceptable.
            //shrink TR
            success = false;
            if(rho_obj_min < 0) {
                delta = delta_decrease_min*delta;
            } else {
                delta = std::max(delta_decrease_min*delta, delta_decrease*arma::norm(s_hat));
            }
            data.record_backtrack(can_rllh);
            data.restore_stencil(); //Do the backtrack so that theta_{n+1} = theta_n
        } else { //Success!
            success = true;
            data.rllh = can_rllh;
            if(rho_obj > rho_obj_opt) { //Great success!
                if(rho_cauchy > rho_obj_opt) { //Backing up was not a big issue
                    //expand TR size for better convergence
                    delta = std::max(delta, delta_increase*arma::norm(s_hat));
                } else if(delta>1 && rho_cauchy <= rho_obj_min) {
                    //s_hat is cauchy point shrink 
                    delta = std::max(delta_decrease*delta, arma::norm(s_hat));
                }
            }
            data.record_iteration();
        }
        
//         std::cout<<"   x_{n+1}:"<<data.theta().t();
//         std::cout<<"   rllh(x_{n+1}):"<<can_rllh<<"\n";
//         std::cout<<"   old_rllh:"<<old_rllh<<"\n";
//         std::cout<<"   model_improvement:"<<model_improvement<<"\n";
//         std::cout<<"   obj_improvement:"<<obj_improvement<<"\n";
//         std::cout<<"   rho_obj:"<<rho_obj<<"\n";
//         std::cout<<"   success:"<<success<<"\n";
//         std::cout<<"   new TrustRadius:"<<delta<<"\n";
        //Test for convergence
        if(delta < 8*this->epsilon) {
            data.record_exit(IterativeMaximizer<Model>::ExitCode::TrustRegionRadius);
            return;
        }
        double grad_ratio = arma::norm(data.grad) / std::fabs(data.rllh);
        if( grad_ratio < this->epsilon) {
            data.record_exit(IterativeMaximizer<Model>::ExitCode::GradRatio);
            return;
        }
        if(success) {
            double step_size_ratio =  (arma::norm(data.theta() - data.saved_theta()) / 
                                        std::max(arma::norm(data.theta()),arma::norm(data.saved_theta())));
            if(step_size_ratio < this->epsilon){
                data.record_exit(IterativeMaximizer<Model>::ExitCode::StepSize);
                return;
            }
        }
    }
    data.record_exit(IterativeMaximizer<Model>::ExitCode::MaxIter);
}

/**
 * This works for either minimization or maximization.  sign(grad2) is not important
 * 
 */
template<class Model>
VecT 
TrustRegionMaximizer<Model>::compute_D_scale(const VecT &oldDscale, const VecT &grad2)
{
    return arma::clamp(arma::max(oldDscale,arma::sqrt(arma::abs(grad2))), min_scaling,max_scaling);
}


/**
 * Works for minimization or maximization.  Indepdendet of sign or grad
 * 
 */
template<class Model>
inline
double 
TrustRegionMaximizer<Model>::compute_initial_trust_radius(const VecT &ghat)
{
    return std::min(std::max(arma::norm(ghat),delta_init_min),delta_init_max);
}

/**
 * @brief Quadratic model value at given step
 * Compute a quadratic model
 */
template<class Model>
inline
double 
TrustRegionMaximizer<Model>::quadratic_model_value(const VecT &s, const VecT &g, const MatT &H)
{
    return arma::dot(g,s) + .5*arma::dot(s, arma::symmatu(H)*s);
}

/**
 * @brief The vector used for bound constrained TR scaling
 * 
 * This v is from Coleman&Li (1996).  It represents a scaling factor for bound constained
 * problems.  For unconstrained problems v = sgn(grad);
 * 
 * In all cases 
 */
template<class Model>
void TrustRegionMaximizer<Model>::compute_bound_scaling_vec(const VecT &theta, const VecT &g,
                                                            const VecT &lbound, const VecT &ubound, 
                                                            VecT &v, VecT &Jv)
{
    int N = static_cast<int>(theta.n_elem);
    v.set_size(N);
    Jv.set_size(N);  //This is the default Jv case if all variables have lower and upper bounds
    for(int i=0; i<N; i++){
        if(g(i)>=0) {
            if(std::isfinite(lbound(i))){
                v(i) = theta(i)-lbound(i);
                Jv(i) = sgn(v(i));
            } else {
                v(i) = 1;
                Jv(i) = 0;
            }
        } else {
            if(std::isfinite(ubound(i))){
                v(i) = theta(i)-ubound(i);
                Jv(i) = sgn(v(i));
            } else {
                v(i) = -1;
                Jv(i) = 0;
            }
        }
    }
}

/**
 * 
 * This is alpha[d] from Coleman and Li
 * 
 * 
 */
template<class Model>
VecT 
TrustRegionMaximizer<Model>::bound_step(const VecT &step_hat, const VecT &D, const VecT &theta, const VecT &lbound, const VecT &ubound)
{
    VecT step = step_hat / D;
    double alpha = arma::min(VecT(arma::max((lbound-theta)/step,(ubound-theta)/step)));
//     std::cout.precision(15);
//     std::cout<<"\n (((BoundStep))) alpha:"<<alpha<<"\n";
//     std::cout<<"   step_hat: "<<step_hat.t();
//     std::cout.precision(15);
//     step.t().raw_print(std::cout,"   step: ");
//     std::cout<<"   D: "<<D.t();
//     std::cout.precision(15);
//     theta.t().raw_print(std::cout,"  Theta:");
//     std::cout<<"   theta in bounds?: "<<model.theta_in_bounds(theta)<<"\n";
//     std::cout<<"   lbound: "<<lbound.t();
//     std::cout<<"   ubound: "<<ubound.t();
//     std::cout.precision(15);
//     (theta+step).t().raw_print(std::cout,"   full_step: ");
    
    if(alpha>1){ //step is feasible.  accept it
//         std::cout<<" (((alpha>1 ==> feasible))) alpha: "<<alpha<<"\n";
        VecT full_step = theta + step;
        if(!arma::all(full_step > lbound)) throw NumericalError("Feasible Bounding failed lower bounds");
        if(!arma::all(full_step < ubound)) throw NumericalError("Feasible Bounding failed upper bounds");
        return step_hat;
    } else { //backtrack a little bit from alpha to remain feasible
        //Bellavia (2004); Coleman and Li (1996)
        double kappa = boundary_stepback_min_kappa;
//         std::cout<<" restricted_step: "<<(theta+alpha*step).t();
//         std::cout<<" kappa: "<<kappa<<"\n";
//         std::cout<<" kappa restricted_step: "<<(theta+kappa*alpha*step).t()<<"\n";
//         std::cout<<" (((alpha<=1 ==> restricted-feasible)))\n";
        VecT full_step = theta + kappa*alpha*step;
        if(!arma::all(full_step > lbound)) throw NumericalError("Infeasible Bounding failed lower bounds");
        if(!arma::all(full_step < ubound)) throw NumericalError("Infeasible Bounding failed upper bounds");
        return kappa*alpha*step_hat; //rescale back to hat space
    }
}


template<class Model>
VecT 
TrustRegionMaximizer<Model>::compute_cauchy_point(const VecT &g, const MatT &H, double delta)
{
    double gnorm = arma::norm(g);
    double Q = arma::dot(g,H*g);
    double tau = (Q<=0) ? 1 : std::min(1.0, gnorm*gnorm*gnorm / (delta*Q));
    return  -(tau*delta/gnorm) * g;
}


/** @brief Exactly solver the TR subproblem even for non-positive definite H
 * 
 * This method is a hybrid technique mixing ideas from
 * Geyer (2013) and the "trust" R-package
 * Nocetal and Wright (2000)
 * More and Sorensen (1981)
 */
template<class Model>
VecT TrustRegionMaximizer<Model>::solve_TR_subproblem(const VecT &g, const MatT &H, double delta, double epsilon)
{
    int N = static_cast<int>(g.n_elem);
    double g_norm = arma::norm(g);
    MatT Hchol=H;
//     copy_Usym_mat(Hchol);
    bool pos_def = cholesky(Hchol);

    if(pos_def) {
        VecT newton_step = cholesky_solve(Hchol,-g);  //this is s(0), i.e., lambda=0
        VecT newton_step2 = arma::solve(arma::symmatu(H),-g);
//         std::cout<<" R:\n"<<Hchol;
//         std::cout<<" newton step:"<<newton_step.t();
//         std::cout<<" newton step2:"<<newton_step2.t();
//         std::cout<<" norm newton step: "<<arma::norm(newton_step)<<"\n";
//         std::cout<<" tr delta: "<<delta<<"\n";
        if(arma::norm(newton_step)<=delta){
            //[Case 1]: Full Newton Step
//             std::cout<<" {{Case 1: Full Newton Step}}";
            return newton_step;
        } else {
            //[Case 2]: Restricted Newton Step
            //Attempt to arrive at bounds that avoids having to do an eigendecomposition in this case
            double lambda_lb = 0;
            double lambda_ub = g_norm/delta;
//             std::cout<<" {{Case 2: Restricted Newton Step}}";
//             std::cout<<"  [lambda range: "<<lambda_lb<<"-"<<lambda_ub<<"]\n";
            return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub, epsilon);
        }
    } else {
        //Indefinite hessian.  Do eigendecomposition to better understand the hessian
        VecT lambda_H;
        MatT Q_H;
        bool decomp_success = arma::eig_sym(lambda_H,Q_H, arma::symmatu(H)); //Compute eigendecomposition of symmertic matrix H
        if(!decomp_success) throw NumericalError("Could not eigendecompose");
        VecT g_hat = Q_H.t() * g; // g in coordinates of H's eigensystem
        double delta2 = delta * delta; //delta^2
        double lambda_min = lambda_H(0); //Minimum eigenvalue.  lambda_H is gaurenteeded in decreaseing order
        
        // compute multiplicity of lambda_min.
        int Nmin = 1; 
        while(lambda_H(Nmin)==lambda_min) Nmin++;
        //Compute ||P_min g'||^2
        double g_min_norm = 0.;
        for(int i=0; i<Nmin;i++) g_min_norm += g_hat(i)*g_hat(i);
        g_min_norm = sqrt(g_min_norm);
//         std::cout<<" {{TR SUBPROBLEM}}\n";
//         std::cout<<" g=["<<g.t()<<"]\n";
//         std::cout<<" H=[\n"<<H<<"]\n";
//         std::cout<<" delta="<<delta<<"\n";
//         std::cout<<" pos_def="<<pos_def<<"\n";
//         std::cout<<"  lambdaH="<<lambda_H.t();
//         std::cout<<"  lambda_min="<<lambda_min<<"\n";
//         std::cout<<"  Nmin="<<Nmin<<"\n";
//         std::cout<<"  g_hat=["<<g_hat.t()<<"]\n";
//         std::cout<<"  g_norm="<<g_norm<<"\n";
//         std::cout<<"  g_min_norm="<<g_min_norm<<"\n";
        if(g_min_norm>0) {
            //[Case 3]: Indefinite hessian, general-position gradiant
            //The gradiant extends into the lambda min subspace, so there is a pole at s(lambda_min)=inf
            double lambda_lb = g_min_norm/delta - lambda_min;
            double lambda_ub = g_norm/delta - lambda_min;
//             std::cout<<" {{Case 3:  Indefinite hessian, general-position gradiant}}";
//             std::cout<<"  lambda range=["<<lambda_lb<<","<<lambda_ub<<"]\n";
            return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub, epsilon);
        } else {
            //Compute s_perp_sq = ||P^perp_min s(lambda_min)||^2
            VecT  s_perp_lmin(N,arma::fill::zeros);
            for(int i=Nmin; i<N; i++) s_perp_lmin += (g_hat(i)/(lambda_H(i)-lambda_min)) * Q_H.col(i);
            double s_perp_lmin_normsq = arma::dot(s_perp_lmin,s_perp_lmin);
            std::cout<<" s_perp_lmin_normsq: "<<s_perp_lmin_normsq<<"\n";
            if(s_perp_lmin_normsq >= delta) {
                //[Case 4]: Indefinite hessian, degenerate gadient, sufficient perpendicular step
                double lambda_lb = -lambda_min;
                double lambda_ub = g_norm/delta - lambda_min;
                std::cout<<" {{Case 4:  Indefinite hessian, degenerate gradient, sufficient perpendicular step}}";
                std::cout<<"  [lambda range: "<<lambda_lb<<"-"<<lambda_ub<<"]\n";
                return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub, epsilon);
            } else {
                //[Case 5]: Indefinite hessian, degenerate gadient, insufficient perpendicular step
                //(i.e., The hard-hard case)
                double tau = sqrt(delta2 - s_perp_lmin_normsq);
                std::cout<<" {{Case 5:  Indefinite hessian, degenerate gradient, insufficient perpendicular step}}";
                std::cout<<" [tau: "<<tau<<" ] q1:"<<Q_H.col(0).t();
                return s_perp_lmin + tau * Q_H.col(0);  // s_perp(lambda_min) + tau * q_min; q_min is vector from S_min subspoace
            }
        }
    }
}

template<class Model>
VecT 
TrustRegionMaximizer<Model>::solve_restricted_step_length_newton(const VecT &g, const MatT &H, double delta, 
                                                                 double lambda_lb, double lambda_ub, double epsilon)
{
    //Initially Hchol is the cholesky decomposition at lambda=0
    double lambda = std::max(lambda_lb,std::min(lambda_ub,std::max(1./delta,.5*(lambda_lb+lambda_ub))));
    int max_iter = 50;
    
    for(int i=0; i<max_iter;i++) {
        MatT R = H;
//         copy_Usym_mat(R);
        R.diag() += lambda;
        bool is_pos = cholesky(R);
//         std::cout<<"* Lambda solve: iter:"<<i<<"\n";
//         std::cout<<"* lambda:"<<lambda<<"\n";
        if(!is_pos) {
            std::cout<<"H:"<<H;
            std::cout<<"lambda: "<<lambda<<"\n";
            throw NumericalError("Bad cholesky decomposition.  Lambda is too small??.");
        }
        VecT p = cholesky_solve(R,-g);
        MatT Rtri = R;
        cholesky_convert_lower_triangular(Rtri);
//         double cond = arma::cond(arma::trimatl(Rtri));
        VecT q = arma::solve(arma::trimatl(Rtri), p);
        double norm_p = arma::norm(p);
        double objective = 1/delta - 1/norm_p;

        if(std::fabs(objective) < epsilon) return p;
        double lambda_delta = (norm_p/arma::norm(q)) * (norm_p/arma::norm(q)) * ((norm_p-delta)/delta);
        if(std::fabs(lambda_delta) < epsilon) return p;
        if(lambda==lambda_lb && lambda_delta<0) {
            VecT lambda_H;
            MatT Q_H;
            arma::eig_sym(lambda_H,Q_H, arma::symmatu(H)); //Compute eigendecomposition of symmertic matrix H
            double lambda_min = lambda_H(0); //Minimum eigenvalue.  lambda_H is gaurenteeded in decreaseing order
            R = H;
//             copy_Usym_mat(R);
            R.diag() += -lambda_min;
            cholesky(R);
            p = cholesky_solve(R,-g);
            Rtri = R;
            cholesky_convert_lower_triangular(Rtri);
            q = arma::solve(arma::trimatl(Rtri), p);
            norm_p = arma::norm(p);
            double objective_lam_min = 1/delta - 1/norm_p;
            std::cout<<" obj(-lambda_min="<<-lambda_min<<")=:"<<objective_lam_min<<"\n";
            std::cout<<" obj(lambda_lb="<<lambda_lb<<")=:"<<objective<<"\n";
            throw NumericalError("Bad lambda lower bound??");
        }
        if(lambda==lambda_ub && lambda_delta>0) {
//             std::cout<<" R:\n"<<R;
//             std::cout<<" Rtri:\n"<<arma::trimatl(Rtri);
//             std::cout<<" cond: "<<cond<<"\n";
//             std::cout<<"* p:"<<p.t();
//             std::cout<<"* q:"<<q.t();
//             std::cout<<"* delta:"<<delta<<"\n";
//             std::cout<<"* |p|:"<<arma::norm(p)<<"\n";
//             std::cout<<"* |q|:"<<arma::norm(q)<<"\n";
//             std::cout<<"* obj:"<<objective<<"\n";
            throw NumericalError("Bad lambda upper bound??");
        }
        lambda += lambda_delta;
        lambda = std::min(std::max(lambda,lambda_lb),lambda_ub);
//         std::cout<<"* lam_delta:"<<lambda_delta<<"\n";
//         std::cout<<"* new_lambda:"<<lambda<<"\n";
        
    }
//     std::cout<<"LAMBDA SEARCH EXCEEDED!!!\n";
//     std::cout<<" g:="<<g;
//     std::cout<<" H:=\n"<<H;
//     std::cout<<" delta:="<<delta<<"\n";
//     std::cout<<" lambda_bnds:= ["<<lambda_lb<<","<<lambda_ub<<"]\n";
//     std::cout<<" epsilon:="<<epsilon<<"\n";
    throw NumericalError("Lambda search exceeded max_iter");
}


/* =========== Simulated Annealing ========== */

template<class Model>
StencilT<Model>
SimulatedAnnealingMaximizer<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh)
{
    MatT sequence;
    VecT sequence_rllh;
    return anneal(im, model.initial_theta_estimate(im,theta_init), rllh, sequence, sequence_rllh);
}

template<class Model>
StencilT<Model>
SimulatedAnnealingMaximizer<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                                           ParamVecT<Model> &sequence, VecT &sequence_rllh)
{
    double rllh;
    return anneal(im, model.initial_theta_estimate(im,theta_init), rllh, sequence, sequence_rllh);
}


template<class Model>
StencilT<Model>
SimulatedAnnealingMaximizer<Model>::anneal(const ModelDataT<Model> &im, const StencilT<Model> &theta_init, 
                                           double &theta_max_rllh, MatT &sequence, VecT &sequence_rllh)
{
    auto &rng = model.get_rng_generator();
    UniformDistT uniform;
    NewtonDiagonalMaximizer<Model> tr_max(model);
//     TrustRegionMaximizer<Model> tr_max(model);
    int niters = max_iterations*model.get_mcmc_num_candidate_sampling_phases();
    sequence = model.make_param_stack(niters+1);
    sequence_rllh.set_size(niters+1);
    sequence.col(0)=theta_init.theta;
    sequence_rllh(0)=methods::objective::rllh(model, im, theta_init);
    double max_rllh = sequence_rllh(0);
    int max_idx = 0;
    StencilT<Model> max_s;
    double T = T_init; //Temperature
    int naccepted=1;
    for(int n=1; n<niters; n++){
        ParamT<Model> can_theta = sequence.col(naccepted-1);
        model.sample_mcmc_candidate_theta(n, can_theta);
        if(!model.theta_in_bounds(can_theta)) { //OOB
            n--;
            continue;
        }
        double can_rllh = methods::objective::rllh(model, im, can_theta);
        double old_rllh = sequence_rllh(naccepted-1);
        if(can_rllh < old_rllh && uniform(rng) > exp((can_rllh-old_rllh)/T)) continue; //Reject
        //Accept
        T /= cooling_rate;
        sequence.col(naccepted) = can_theta;
        sequence_rllh(naccepted) = can_rllh;
        if(can_rllh > max_rllh) {
            max_rllh = can_rllh;
            max_idx = naccepted;
        }
        naccepted++;
        if(naccepted >= niters+1) throw LogicalError("Iteration error in simulated annealing");
    }

    //Run a  maximization
    tr_max.local_maximize(im, model.make_stencil(sequence.col(max_idx)), max_s, max_rllh);
    //Fixup sequence to return
    sequence.resize(sequence.n_rows, naccepted+1);
    sequence.col(naccepted) = max_s.theta;
    sequence_rllh.resize(naccepted+1);
    sequence_rllh(naccepted) = max_rllh;
    theta_max_rllh = max_rllh;
    return max_s;
}

} /* namespace mappel */

#endif /* _MAPPEL_ESTIMATOR_IMPL_H */
