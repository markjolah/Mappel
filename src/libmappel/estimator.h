/** @file estimator.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
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
#include <map>
#include "rng.h"
#include "cGaussMLE/cGaussMLE.h"

#ifdef WIN32
    #include <boost/chrono.hpp>
    typedef boost::chrono::high_resolution_clock ClockT;
#else
    #include <chrono>
    typedef std::chrono::high_resolution_clock ClockT;
#endif

#include "util.h"

namespace mappel {

static const int DEFAULT_ITERATIONS=100;
static const int DEFAULT_CGAUSS_ITERATIONS=50;


template<class Model>
class Estimator{
public:
    /* These improve readability, but are (unfortunately) not inherited. */
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    using ModelDataStackT = typename Model::ModelDataStackT;

    Model &model;

    Estimator(Model &_model) : model(_model) {}
    virtual ~Estimator() {}

    virtual std::string name() const =0;

    /* These are the main entrypoints for estimation
     * We provide three types of interfaces:
     */
    /* Option 1: Single Image - Estimator returns a stencil at the optimum point.
     * Variants with and without a theta_init or a rllh output parameter
     */
    StencilT estimate_max(const ModelDataT &im);
    StencilT estimate_max(const ModelDataT &im, const ParamT &theta_init); //with theta_init
    StencilT estimate_max(const ModelDataT &im, double &rllh); //with rllh out
    StencilT estimate_max(const ModelDataT &im, const ParamT &theta_init, double &rllh); //with theta_init and rllh out

    /* Option 2: Single Image - Estimator returns theta,  rllh and obsI  at the optimum point.
     * Variants with and without a theta_init.
     */
    void estimate_max(const ModelDataT &im, ParamT &theta, double &rllh, MatT &obsI);
    void estimate_max(const ModelDataT &im, const ParamT &theta_init, ParamT &theta, double &rllh, MatT &obsI);//with theta_init

    /* Option 3: Single Image Debug mode - Estimator returns theta, rllh and obsI   at the optimum point.
     * Variants with and without a theta_init.
     */
    void estimate_max_debug(const ModelDataT &im, const ParamT &theta_init, 
                        ParamT &theta_est, MatT &obsI, MatT &sequence, VecT &sequence_rllh);

    /* Option 4: Parallel Image - Estimator returns theta_stack, rllh_stack and obsI_stack at the optimum point
     * for each image in the stack.
     * Variants with and without a theta_init.
     */
    virtual void estimate_max_stack(const ModelDataStackT &im_stack, const ParamVecT &theta_init_stack, 
                                ParamVecT &theta_est_stack, VecT &rllh_stack, CubeT &obsI_stack)=0;
    void estimate_max_stack(const ModelDataStackT &im_stack, 
                                ParamVecT &theta_est_stack, VecT &rllh_stack, CubeT &obsI_stack);

    /* Statistics */
    virtual StatsT get_stats();
    virtual StatsT get_debug_stats()=0;
    virtual void clear_stats();

    /* I/O */
    template<class T>
    friend std::ostream& operator<<(std::ostream &out, Estimator<T> &estimator);

protected:
    virtual StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh)=0;
    virtual StencilT compute_estimate_debug(const ModelDataT &im, const ParamT &theta_init, 
                                            ParamVecT &sequence, VecT &sequence_rllh);
    virtual void compute_estimate(const ModelDataT &im, const ParamT &theta_init, ParamT &theta_est, double &rllh, MatT &obsI);

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
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataStackT = typename Model::ModelDataStackT;

    ThreadedEstimator(Model &model);

    void estimate_max_stack(const ModelDataStackT &im, const ParamVecT &theta_init,ParamVecT &theta, VecT &rllh, CubeT &obsI);
    
    StatsT get_stats();
    StatsT get_debug_stats();
    void clear_stats();

protected:
    using Estimator<Model>::model;
    int max_threads;
    int num_threads;
    boost::mutex mtx;
};

template<class Model>
class HeuristicEstimator : public ThreadedEstimator<Model> {
public:
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ModelDataT = typename Model::ModelDataT;
    HeuristicEstimator(Model &model) : ThreadedEstimator<Model>(model) {}

    std::string name() const {return "HeuristicEstimator";}
private:
    StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh);
};

template<class Model>
class CGaussHeuristicEstimator : public ThreadedEstimator<Model> {
public:
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ModelDataT = typename Model::ModelDataT;
    CGaussHeuristicEstimator(Model &model) : ThreadedEstimator<Model>(model) {}
    
    std::string name() const {return "CGaussHeuristicEstimator";}
private:
    using Estimator<Model>::model;
    inline
    StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh);
};


template<class Model>
class CGaussMLE : public ThreadedEstimator<Model> {
public:
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    int max_iterations;
    CGaussMLE(Model &model, int max_iterations=DEFAULT_CGAUSS_ITERATIONS)
        : ThreadedEstimator<Model>(model), max_iterations(max_iterations) {}

    StatsT get_stats();
    StatsT get_debug_stats();

    std::string name() const {return "CGaussMLE";}
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;

    StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh);
    StencilT compute_estimate_debug(const ModelDataT &im, const ParamT &theta_init, 
                                    ParamVecT &sequence, VecT &sequence_rllh);
};


template<class Model>
class SimulatedAnnealingMaximizer : public ThreadedEstimator<Model> {
public:
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    
    using Estimator<Model>::model;

    double T_init=100.;
    double cooling_rate=1.02;
    double max_iterations=500;

    inline std::string name() const {return "SimulatedAnnealingMaximizer";}
    SimulatedAnnealingMaximizer(Model &model) : ThreadedEstimator<Model>(model) {}
protected:
    StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh);
    StencilT compute_estimate_debug(const ModelDataT &im, const ParamT &theta_init, 
                                    ParamVecT &sequence, VecT &sequence_rllh);
    StencilT anneal(const ModelDataT &im, const StencilT &theta_init,
                    double &rllh, ParamVecT &sequence, VecT &sequence_rllh);
};

template<class Model>
class IterativeMaximizer : public ThreadedEstimator<Model> {
public:
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    
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
    void local_maximize(const ModelDataT &im, const StencilT &theta_init, StencilT &stencil, double &rllh); //This is used by SimulatedAnnealing to clean up max

protected:
    using Estimator<Model>::model;
    using Estimator<Model>::num_estimations;
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
    /* Debug Statistics: Only active in debug mode when data.save_seq==true */
    IdxVecT last_backtrack_idxs;

    class MaximizerData {
    public:
        using StencilT = typename Model::Stencil;
        using ParamT = typename Model::ParamT;
        using ParamVecT = typename Model::ParamVecT;
        using ModelDataT = typename Model::ModelDataT;
        const ModelDataT &im;
        ParamT grad;
        ParamT step;
        VecT lbound, ubound;
        double rllh;
        int nBacktracks=0;
        int nIterations=0;
        bool save_seq;
        
        MaximizerData(const Model &model, const ModelDataT &im, const StencilT &s,
                      bool save_seq=false, int max_seq_len=0);

        /** @brief Record an iteration point (derivatives computed) Using the saved theta as the default. */
        void record_iteration() {record_iteration(theta());}
        /** @brief Record an iteration point (derivatives computed) */
        void record_iteration(const ParamT &accepted_theta);
        /** @brief Record a backtracked point (no derivative computations performed) Using the saved theta as the default. */
        void record_backtrack(double rejected_rllh) {record_backtrack(theta(), rejected_rllh);}
        /** @brief Record a backtracked point (no derivative computations performed) */
        void record_backtrack(const ParamT &rejected_theta, double rejected_rllh);
        
        /** @brief Return the saved theta sequence */
        ParamVecT get_theta_sequence() const {return theta_seq.head_cols(seq_len);}
        IdxVecT get_backtrack_idxs() const {return backtrack_idxs.head(seq_len);}
        VecT get_theta_sequence_rllh() const {return seq_rllh.head(seq_len);}
        /** @brief Get the current stencil  */
        StencilT& stencil() {return current_stencil ? s0 : s1;}
        void set_stencil(const StencilT &s) {if(current_stencil) s0=s;  else s1=s; }
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
        StencilT& saved_stencil() {return current_stencil ? s1 : s0;}
        /** @brief Get the current stencil's theta  */
        ParamT& theta() {return current_stencil ? s0.theta : s1.theta;}
        /** @brief Get the saved stencil's theta  */
        ParamT& saved_theta() {return current_stencil ? s1.theta : s0.theta;}
        int getIteration() const {return seq_len;}
    protected:
        StencilT s0,s1; //These two stencils will be alternated as the current and old stencil points
        bool current_stencil; //This alternates to indicated weather s0 or s1 is the current stencil

        ParamVecT theta_seq;
        VecT seq_rllh;
        IdxVecT backtrack_idxs;
        int seq_len=0;
        const int max_seq_len;
    };

    void record_run_statistics(const MaximizerData &data);

    StencilT compute_estimate(const ModelDataT &im, const ParamT &theta_init, double &rllh);
    StencilT compute_estimate_debug(const ModelDataT &im, const ParamT &theta_init, 
                                    ParamVecT &sequence, VecT &sequence_rllh);

    virtual void maximize(MaximizerData &data)=0;
    inline bool backtrack(MaximizerData &data);
    inline bool convergence_test(MaximizerData &data);
};



template<class Model>
class NewtonDiagonalMaximizer : public IterativeMaximizer<Model> {
public:
     /* These improve readability, but are (unfortunately) not inherited. */
     using StencilT = typename Model::Stencil;
     using ParamT = typename Model::ParamT;
     using ParamVecT = typename Model::ParamVecT;
     using ModelDataT = typename Model::ModelDataT;
    typedef typename IterativeMaximizer<Model>::MaximizerData MaximizerData;
    

    NewtonDiagonalMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonDiagonalMaximizer";}
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;
    using IterativeMaximizer<Model>::record_run_statistics;
    using IterativeMaximizer<Model>::max_iterations;
    using IterativeMaximizer<Model>::backtrack;
    using IterativeMaximizer<Model>::convergence_test;

    void maximize(MaximizerData &data);
};

template<class Model>
class NewtonMaximizer : public IterativeMaximizer<Model> {
public:
     /* These improve readability, but are (unfortunately) not inherited. */
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    using ModelDataStackT = typename Model::ModelDataStackT;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;


    NewtonMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonMaximizer";}
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;
    using IterativeMaximizer<Model>::record_run_statistics;
    using IterativeMaximizer<Model>::max_iterations;
    using IterativeMaximizer<Model>::backtrack;
    using IterativeMaximizer<Model>::convergence_test;


    void maximize(MaximizerData &data);
};

template<class Model>
class QuasiNewtonMaximizer : public IterativeMaximizer<Model> {
public:
     /* These improve readability, but are (unfortunately) not inherited. */
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    using ModelDataStackT = typename Model::ModelDataStackT;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;



    QuasiNewtonMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "QuasiNewtonMaximizer";}
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;
    using IterativeMaximizer<Model>::record_run_statistics;
    using IterativeMaximizer<Model>::max_iterations;
    using IterativeMaximizer<Model>::backtrack;
    using IterativeMaximizer<Model>::convergence_test;


    void maximize(MaximizerData &data);
};

template<class Model>
class TrustRegionMaximizer : public IterativeMaximizer<Model> {
public:
    /* These improve readability, but are (unfortunately) not inherited. */
    using StencilT = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ModelDataT = typename Model::ModelDataT;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;
    
    static constexpr double rho_cauchy_min = 0.1;  //Coleman beta | Bellavia beta_1
    static constexpr double rho_obj_min = 0.25;  //Coleman mu | Bellavia beta_2
    static constexpr double rho_obj_opt = 0.75;  //Coleman eta | Bellavia beta_2
    static constexpr double delta_decrease_min = 0.125; // Coleman gamma_0 | Bellavia alpha_1
    static constexpr double delta_decrease = 0.25; // Coleman gamma_1 | Bellavia alpha_2
    static constexpr double delta_increase = 2; // Coleman gamma_2 | Bellavia alpha_3
    
    static constexpr double min_scaling = 1.0e-5;  //Minimum for Dscale(i)
    static constexpr double max_scaling = 1.0e5;   //Maximum for Dscale(i)
    static constexpr double delta_init_min = 1.0e-3; //Minimum initial trust region radius
    static constexpr double delta_init_max = 1.0e3;  //Maximum initial trust region radius
    static constexpr double boundary_stepback_min_kappa = 1.0 - 1.0e-5; //Distance to step back from the bounadary to remain in interrior
    
    TrustRegionMaximizer(Model &model, int max_iterations=DEFAULT_ITERATIONS)
        : IterativeMaximizer<Model>(model,max_iterations) {}
    
    inline std::string name() const {return "TrustRegionMaximizer";}
    
protected:
    /* These bring in non-depended names from base classes (only necessary because we are templated) */
    using Estimator<Model>::model;
    using IterativeMaximizer<Model>::record_run_statistics;
    using IterativeMaximizer<Model>::max_iterations;
    using IterativeMaximizer<Model>::convergence_test;
    
    void maximize(MaximizerData &data);
    
    static VecT compute_D_scale(const VecT &oldDscale, const VecT &grad2);
    static double compute_initial_trust_radius(const VecT &ghat);
    static double quadratic_model_value(const VecT &step, const VecT &grad, const MatT &hess);

    static void compute_bound_scaling_vec(const VecT &theta, const VecT &grad, const VecT &lbound, const VecT &ubound, VecT &v, VecT &Jv);
     VecT bound_step(const VecT &step_hat, const VecT &D, const VecT &theta, const VecT &lbound, const VecT &ubound);
    static VecT compute_cauchy_point(const VecT &g, const MatT& H, double delta);
    static VecT solve_TR_subproblem(const VecT &g, const MatT &H, double delta, double epsilon);
    static VecT solve_restricted_step_length_newton(const VecT &g, const MatT &H, double delta, double lambda_lb, double lambda_ub, double epsilon);
};


} /* namespace mappel */

#endif /* _ESTIMATOR_H */
