/** @file estimator.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for the 
 * Estimator class hierarchy.
 */
#ifndef _ESTIMATOR_H
#define _ESTIMATOR_H

#include <exception>
#include <fstream>
#include <string>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include "Mappel/rng.h"
#include "cGaussMLE/cGaussMLE.h"

#ifdef WIN32
    #include <boost/chrono.hpp>
    typedef boost::chrono::high_resolution_clock ClockT;
#else
    #include <chrono>
    typedef std::chrono::high_resolution_clock ClockT;
#endif

#include "Mappel/util.h"

namespace mappel {

static const int DEFAULT_ITERATIONS=100;
static const int DEFAULT_CGAUSS_ITERATIONS=50;


template<class Model>
class Estimator{
public:
    Estimator(Model &_model) : model(_model) {}
    virtual ~Estimator() {}

    virtual std::string name() const =0;
    Model& get_model();
    void set_model(Model &new_model);

    /* These are the main entrypoints for estimation
     * We provide three types of interfaces:
     */
    /* Option 1: Single Image - Estimator returns a stencil at the optimum point.
     * Variants with and without a theta_init or a rllh output parameter
     */
    StencilT<Model> estimate_max(const ModelDataT<Model> &im);
    StencilT<Model> estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init); //with theta_init
    StencilT<Model> estimate_max(const ModelDataT<Model> &im, double &rllh); //with rllh out
    StencilT<Model> estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh); //with theta_init and rllh out

    /* Option 2: Single Image - Estimator returns theta,  rllh and obsI  at the optimum point.
     * Variants with and without a theta_init.
     */
    void estimate_max(const ModelDataT<Model> &im, ParamT<Model> &theta, double &rllh, MatT &obsI);
    void estimate_max(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, ParamT<Model> &theta, double &rllh, MatT &obsI);//with theta_init

    /* Option 3: Single Image Debug mode - Estimator returns theta, rllh and obsI   at the optimum point.
     * Variants with and without a theta_init.
     */
    void estimate_max_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                        ParamT<Model> &theta_est, double &rllh,MatT &obsI, MatT &sequence, VecT &sequence_rllh);

    /* Option 4: Parallel Image - Estimator returns theta_stack, rllh_stack and obsI_stack at the optimum point
     * for each image in the stack.
     * Variants with and without a theta_init.
     */
    virtual void estimate_max_stack(const ModelDataStackT<Model> &im_stack, const ParamVecT<Model> &theta_init_stack, 
                                ParamVecT<Model> &theta_est_stack, VecT &rllh_stack, CubeT &obsI_stack)=0;
    void estimate_max_stack(const ModelDataStackT<Model> &im_stack, 
                                ParamVecT<Model> &theta_est_stack, VecT &rllh_stack, CubeT &obsI_stack);

    virtual void estimate_profile_stack(const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &values, const ParamVecT<Model> &theta_init,
                                        VecT &profile_likelihood, ParamVecT<Model> &profile_parameters)=0;

    /* Statistics */
    virtual StatsT get_stats();
    virtual StatsT get_debug_stats()=0;
    virtual void clear_stats();

    /* I/O */
    template<class T>
    friend std::ostream& operator<<(std::ostream &out, Estimator<T> &estimator);

protected:
    virtual StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh)=0;
    virtual StencilT<Model> compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                            ParamVecT<Model> &sequence, VecT &sequence_rllh);
    virtual void compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, ParamT<Model> &theta_est, double &rllh, MatT &obsI);
    virtual void compute_profile_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, const IdxVecT &fixed_parameters, ParamT<Model> &theta_est, double &rllh);

    Model &model;

    /* statistics */
    int num_estimations = 0;
    double total_walltime = 0.;

    void record_walltime(ClockT::time_point start_walltime, int nimages);
};

/**
 * We avoid combining Estimator and ThreadedEstimator classes so that a future GPU implementation can
 * inherit directly from Estimator as it will present a differnt method for estimate_stack pure virtual
 * member function.  For now all other (CPU) estimators inherit from ThreadedEstimator.
 *
 */
template<class Model>
class ThreadedEstimator : public Estimator<Model> {
public:
    ThreadedEstimator(Model &model);

    void estimate_max_stack(const ModelDataStackT<Model> &im, const ParamVecT<Model> &theta_init,ParamVecT<Model> &theta, VecT &rllh, CubeT &obsI);
    void estimate_profile_stack(const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &values, const ParamVecT<Model> &theta_init,
                                 VecT &profile_likelihood, ParamVecT<Model> &profile_parameters);

    StatsT get_stats();
    StatsT get_debug_stats();
    void clear_stats();

protected:
    using Estimator<Model>::model;
    int max_threads;
    int num_threads;
    std::mutex mtx;
};

template<class Model>
class HeuristicEstimator : public ThreadedEstimator<Model> {
public:
    HeuristicEstimator(Model &model) : ThreadedEstimator<Model>(model) {}

    std::string name() const {return "HeuristicEstimator";}
private:
    StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh);
};

template<class Model>
class CGaussHeuristicEstimator : public ThreadedEstimator<Model> {
public:
    CGaussHeuristicEstimator(Model &model) : ThreadedEstimator<Model>(model) {}
    
    std::string name() const {return "CGaussHeuristicEstimator";}
private:
    using Estimator<Model>::model;
    StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh);
};


template<class Model>
class CGaussMLE : public ThreadedEstimator<Model> {
public:
    int max_iterations;
    CGaussMLE(Model &model, int max_iterations=DEFAULT_CGAUSS_ITERATIONS)
        : ThreadedEstimator<Model>(model), max_iterations(max_iterations) {}

    StatsT get_stats();
    StatsT get_debug_stats();

    std::string name() const {return "CGaussMLE";}
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;

    StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh);
    StencilT<Model> compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                    ParamVecT<Model> &sequence, VecT &sequence_rllh);
};


template<class Model>
class SimulatedAnnealingMaximizer : public ThreadedEstimator<Model> {
public:    
    double T_init=100.;
    double cooling_rate=1.02;
    int max_iterations=500;

    std::string name() const {return "SimulatedAnnealingMaximizer";}
    SimulatedAnnealingMaximizer(Model &model) : ThreadedEstimator<Model>(model) {}
protected:
    using Estimator<Model>::model;

    StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh);
    StencilT<Model> compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                    ParamVecT<Model> &sequence, VecT &sequence_rllh);
    StencilT<Model> anneal(const ModelDataT<Model> &im, const StencilT<Model> &theta_init,
                    double &rllh, MatT &sequence, VecT &sequence_rllh);
};

template<class Model>
class IterativeMaximizer : public ThreadedEstimator<Model> {
public:
    static constexpr int NumExitCodes = 7;
    enum class ExitCode : IdxT { Unassigned= 99, //Logical error if this is still set
                          MaxIter = 6,        //Max iterations exceeded. Did not converge.
                          MaxBacktracks = 5,  //Backtracking failed.  Likely converged successfully.
                          TrustRegionRadius = 4,//Trust region size was less than epsilon.  Converged successfully.
                          GradRatio = 3,      //Grad ratio was less than epsilon.  Converged successfully.
                          FunctionChange = 2, //Function value change was less than epsilon.  Converged successfully.
                          StepSize = 1,       //Step size was less than delta.  Converged successfully. 
                          Error = 0           //A Numerical Error was caught.  Did not converge.
                        }; 

    IterativeMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS);

    /* Statistics */
    double mean_iterations();
    double mean_backtracks();
    double mean_fun_evals();
    double mean_der_evals();
    StatsT get_stats();
    StatsT get_debug_stats();
    void clear_stats();
    
    /** @brief Perform a local maximization to finish off a simulated annealing run */
    void local_maximize(const ModelDataT<Model> &im, const StencilT<Model> &theta_init, StencilT<Model> &stencil, double &rllh); //This is used by SimulatedAnnealing to clean up max

protected:
    using Estimator<Model>::model;
    using ThreadedEstimator<Model>::mtx;
    int max_iterations;

    /* These parameters control the adaptive convergence testing */
    double epsilon = sqrt(std::numeric_limits<double>::epsilon()); //tolerance for fval
    double delta = sqrt(std::numeric_limits<double>::epsilon()); // tolerance for relative step size
    /* These parameters control backtracking */
    double lambda_min = 0.05; //What is the minimum proportion of the step to take
    double alpha = 1e-4; //How much drop in f-val do we expect for the step to be OK?
    int max_backtracks = 8; //Max # of evaluations to do when backtracking
    
    /* Statistics: need to be private so they can be mutex protected */
    int total_iterations = 0;
    int total_backtracks = 0;
    int total_fun_evals = 0;
    int total_der_evals = 0;
    IdxVecT exit_counts;
    /* Debug Statistics: Only active in debug mode when data.save_seq==true */
    IdxVecT last_backtrack_idxs;

    class MaximizerData {
    public:
        const ModelDataT<Model> &im;
        ParamT<Model> grad;
        ParamT<Model> step;
        VecT lbound, ubound;
        double rllh;
        int nBacktracks=0;
        int nIterations=0;
        bool save_seq;
        ExitCode exit_code=ExitCode::Unassigned;
        MaximizerData(const Model &model, const ModelDataT<Model> &im, const StencilT<Model> &s,
                      bool save_seq=false, int max_seq_len=0);

        void record_exit(ExitCode code);
        /** @brief Record an iteration point (derivatives computed) Using the saved theta as the default. */
        void record_iteration() {record_iteration(theta());}
        /** @brief Record an iteration point (derivatives computed) */
        void record_iteration(const ParamT<Model> &accepted_theta);
        /** @brief Record a backtracked point (no derivative computations performed) Using the saved theta as the default. */
        void record_backtrack(double rejected_rllh) {record_backtrack(theta(), rejected_rllh);}
        /** @brief Record a backtracked point (no derivative computations performed) */
        void record_backtrack(const ParamT<Model> &rejected_theta, double rejected_rllh);
        
        /** @brief Return the saved theta sequence */
        ParamVecT<Model> get_theta_sequence() const {return theta_seq.head_cols(seq_len);}
        IdxVecT get_backtrack_idxs() const {return backtrack_idxs.head(seq_len);}
        VecT get_theta_sequence_rllh() const {return seq_rllh.head(seq_len);}
        /** @brief Get the current stencil  */
        StencilT<Model>& stencil() {return current_stencil ? s0 : s1;}
        void set_stencil(const StencilT<Model> &s) {if(current_stencil) s0=s;  else s1=s; }
        /** @brief Save the current stencil to the single reserve spot.  
         * Overwrites any previously saved stencil.
         * This is used  to save a stencil when backtracking.
         */
        void save_stencil() {current_stencil = !current_stencil;}
        /** @brief Restore the single reserved stencil to the current stencil spot.  
         * Overwrites any previously saved stencil.
         * This is used to restore a last good iterate (and associated stencil data) when backtracking.
         */
        void restore_stencil() {current_stencil = !current_stencil;}
        /** @brief Get the saved stencil  */
        StencilT<Model>& saved_stencil() {return current_stencil ? s1 : s0;}
        /** @brief Get the current stencil's theta  */
        ParamT<Model>& theta() {return current_stencil ? s0.theta : s1.theta;}
        /** @brief Get the saved stencil's theta  */
        ParamT<Model>& saved_theta() {return current_stencil ? s1.theta : s0.theta;}
        int getIteration() const {return seq_len;}
        void set_fixed_parameters(const IdxVecT &fixed_parameters);
        VecT fixed_parameter_scalar;
        bool has_fixed_parameters=false; //True for profile likelihood maximization
    protected:

        StencilT<Model> s0,s1; //These two stencils will be alternated as the current and old stencil points
        bool current_stencil; //This alternates to indicated weather s0 or s1 is the current stencil

        ParamVecT<Model> theta_seq;
        VecT seq_rllh;
        IdxVecT backtrack_idxs;
        int seq_len=0;
        const int max_seq_len;

    };

    void record_run_statistics(const MaximizerData &data);

    StencilT<Model> compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, double &rllh);
    StencilT<Model> compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, 
                                    ParamVecT<Model> &sequence, VecT &sequence_rllh);
    void compute_profile_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, const IdxVecT& fixed_parameters, ParamT<Model> &theta_est, double &rllh);
    virtual void maximize(MaximizerData &data)=0;
    bool backtrack(MaximizerData &data);
    bool convergence_test(MaximizerData &data);
};



template<class Model>
class NewtonDiagonalMaximizer : public IterativeMaximizer<Model> {
public:
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;
    
    NewtonDiagonalMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonDiagonalMaximizer";}
protected:
    using Estimator<Model>::model;
    void maximize(MaximizerData &data);
};

template<class Model>
class NewtonMaximizer : public IterativeMaximizer<Model> {
public:
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    NewtonMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonMaximizer";}
protected:
    using Estimator<Model>::model;
    void maximize(MaximizerData &data);
};

template<class Model>
class QuasiNewtonMaximizer : public IterativeMaximizer<Model> {
public:
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    QuasiNewtonMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "QuasiNewtonMaximizer";}
protected:
    using Estimator<Model>::model;
    void maximize(MaximizerData &data);
};

template<class Model>
class TrustRegionMaximizer : public IterativeMaximizer<Model> {
public:
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;
    
    static const double rho_cauchy_min;// = 0.1;  //Coleman beta | Bellavia beta_1
    static const double rho_obj_min;// = 0.25;  //Coleman mu | Bellavia beta_2
    static const double rho_obj_opt;// = 0.75;  //Coleman eta | Bellavia beta_2
    static const double delta_decrease_min;// = 0.125; // Coleman gamma_0 | Bellavia alpha_1
    static const double delta_decrease;// = 0.25; // Coleman gamma_1 | Bellavia alpha_2
    static const double delta_increase;// = 2; // Coleman gamma_2 | Bellavia alpha_3
    
    static const double min_scaling;// = 1.0e-5;  //Minimum for Dscale(i)
    static const double max_scaling;// = 1.0e5;   //Maximum for Dscale(i)
    static const double delta_init_min;// = 1.0e-3; //Minimum initial trust region radius
    static const double delta_init_max;// = 1.0e3;  //Maximum initial trust region radius
    static const double boundary_stepback_min_kappa;// = 1.0 - 1.0e-5; //Distance to step back from the bounadary to remain in interrior
    
    TrustRegionMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}
    
    inline std::string name() const {return "TrustRegionMaximizer";}
    
protected:        
    using Estimator<Model>::model;
    void maximize(MaximizerData &data);
    VecT bound_step(const VecT &step_hat, const VecT &D, const VecT &theta, const VecT &lbound, const VecT &ubound);
    
    static VecT compute_D_scale(const VecT &oldDscale, const VecT &grad2);
    static double compute_initial_trust_radius(const VecT &ghat);
    static double quadratic_model_value(const VecT &step, const VecT &grad, const MatT &hess);

    static void compute_bound_scaling_vec(const VecT &theta, const VecT &grad, const VecT &lbound, const VecT &ubound, VecT &v, VecT &Jv);
    static VecT compute_cauchy_point(const VecT &g, const MatT& H, double delta);
    static VecT solve_TR_subproblem(const VecT &g, const MatT &H, double delta, double epsilon);
    static VecT solve_restricted_step_length_newton(const VecT &g, const MatT &H, double delta, double lambda_lb, double lambda_ub, double epsilon);
};


} /* namespace mappel */

#endif /* _ESTIMATOR_H */
