/** @file estimator.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for the 
 * Estimator class hierarchy.
 */
#ifndef MAPPEL_ESTIMATOR_H
#define MAPPEL_ESTIMATOR_H

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

namespace estimator {

/** Data reporting structures
 */
///@{
/** A maximum-likelihood estimate for a single image.
 * A container to group the necessary information at an MLEstimate
 */
struct MLEData {
    VecT theta;  ///< Theta estimate
    double rllh; ///< RLLH at theta
    MatT obsI; ///< Observed Fisher information matrix at theta
};


/** A maximum-likelihood estimate for a single image with debugging information.
 * A container to group the necessary information at an MLEstimate
 */
struct MLEDebugData
{
    VecT theta; ///< Theta estimate
    double rllh; ///< RLLH at theta
    MatT obsI; ///< Observed Fisher information matrix at theta
    IdxT Nseq; ///< Number of points evaluated including theta_init and theta_mle.
    MatT sequence; ///< Sequence of evaluated points including theta_init and theta_mle.
    VecT sequence_rllh; ///< RLLH at each point in sequence

    MLEData makeMLEData() const;
};

/** A stack of maximum-likelihood estimates for a stack of images
 * A container to group the necessary information at an MLEstimate
 */
struct MLEDataStack
{
    IdxT Ndata;  ///< Number of data estimates
    MatT theta; ///< Theta estimate stack. size:[Nparams,Ndata]
    VecT rllh; ///< RLLH stack. size:[Ndata]
    CubeT obsI;///< Observed Fisher information matrix stack. size:[Nparams,Nparams,Ndata]
};

/** Container for profile liklihood estimator data
 * Includes both controlling (input) parameters as well as reporting (ouptut) parameters to give output parameters context.
 */
struct ProfileLikelihoodData
{
    /* Input parameters */
    IdxVecT fixed_idxs; ///< Indexes of fixed parameters
    MatT fixed_values; ///< Vector values for each fixed parameter size:[Nfixed,Nvalues];

    /* Output parameters */
    IdxT Nfixed=0; ///< Number of fixed parameters
    IdxT Nvalues=0; ///< Number of values of fixed parameters evaluated
    VecT profile_likelihood; ///< profile likelhood for each column of fixed parameter values
    MatT profile_parameters; ///< Points at which the profile liklihood maximum was obtained.
};

/** Data related to a profile bounds estimation for a single image
 * Includes both controlling (input) parameters as well as reporting (ouptut) parameters to give output parameters context.
 */
struct ProfileBoundsData {

    /* input parameters */
    IdxVecT estimated_idxs; ///< List of indexs for computed parameters.  Empty to compute all parameters.
    double confidence=-1; ///< Confidence level.  If invalid, use default value.
    MLEData mle; ///< Theta maximum-likelihood estimate, rllh, and ObsI

    /* output parameters */
    double target_rllh_delta=-INFINITY; ///< Targeted rllh change in value from MLE ( -chi2inv(confidence,1)/2 )
    IdxT Nparams_est=0; ///< number of parameters estimated =estimated_param_idxs.n_elem.
    VecT profile_lb; ///< size:[Nparams_est] Lower bound estimated at each estimated_idx.
    VecT profile_ub; ///< size:[Nparams_est] Upper bound estimated at each estimated_idx.
    MatT profile_points_lb; ///< size:[NumParams,Nparams_est] Optimal theta found at each lower bound estimate for each estimated_idx.
    MatT profile_points_ub; ///< size:[NumParams,Nparams_est] Optimal theta found at each upper bound estimate for each estimated_idx.
    VecT profile_points_lb_rllh; ///< size:[Nparams_est] RLLH at each of the profile_points_lb
    VecT profile_points_ub_rllh; ///< size:[Nparams_est] RLLH at each of the profile_points_lb

    void initialize_arrays(IdxT Nparams);
};

/** Data for debugging of estimation of profile bounds for a single parameter of a single image
 * Includes both controlling (input) parameters as well as reporting (ouptut) parameters to give output parameters context.
 */
struct ProfileBoundsDebugData {
    /* input parameters */
    IdxT estimated_idx=0; ///< Index of single parameter to estimate for
    MLEData mle; ///< Theta maximum-likelihood estimate, rllh, and ObsI
    double target_rllh_delta=-INFINITY; ///< Targeted rllh change in value from MLE ( -chi2inv(confidence,1)/2 )

    /* output parameters */
    double profile_lb; ///< size:[Nparams_est] Lower bound estimated for estimated_idx.
    double profile_ub; ///< size:[Nparams_est] Upper bound estimated for estimated_idx.
    IdxT Nseq_lb; ///< Number of points in sequence_lb
    IdxT Nseq_ub; ///< Number of points in sequence_ub
    MatT sequence_lb; ///< size:[NumParams,Nseq_lb] Sequence of evaluated points for lb estimate (including theta mle as initial point)
    MatT sequence_ub; ///< size:[NumParams,Nseq_ub] Sequence of evaluated points for ub estimate (including theta mle as initial point)
    VecT sequence_lb_rllh; ///< size:[Nseq_lb] RLLH at each of the sequence_lb points
    VecT sequence_ub_rllh; ///< size:[Nseq_ub] RLLH at each of the sequence_ub points
};


/** Data related to a profile bounds estimation for a stack of images
 * Includes both controlling (input) parameters as well as reporting (ouptut) parameters to give output parameters context.
 */
struct ProfileBoundsDataStack {
    /* input parameters */
    IdxVecT estimated_idxs; ///< List of indexs for computed parameters.  Empty to compute all parameters.
    double confidence=-1; ///< Confidence level.  If invalid, use default value.
    MLEDataStack mle; ///< Theta maximum-likelihood estimate, rllh, and ObsI stack

    /* output parameters */
    IdxT Nparams_est=0; ///< number of parameters estimated =estimated_param_idxs.n_elem.
    IdxT Ndata=0; ///< size of the data stack estimated. (number of individual problem data estimates performed.)
    double target_rllh_delta=-INFINITY; ///< Targeted rllh change in value from MLE ( -chi2inv(confidence,1)/2 )
    MatT profile_lb; ///< size:[Nparams_est,Ndata] Lower bound estimated at each estimated_idx.
    MatT profile_ub; ///< size:[Nparams_est,Ndata] Upper bound estimated at each estimated_idx.
    CubeT profile_points_lb; ///< size:[Nparams,Nparams_est,Ndata] Optimal theta found at each lower bound estimate for each estimated_idx.
    CubeT profile_points_ub; ///< size:[Nparams,Nparams_est,Ndata] Optimal theta found at each upper bound estimate for each estimated_idx.
    MatT profile_points_lb_rllh; ///< size:[Nparams_est,Ndata] RLLH at each of the profile_points_lb
    MatT profile_points_ub_rllh; ///< size:[Nparams_est,Ndata] RLLH at each of the profile_points_ub
    void initialize_arrays(IdxT Nparams);
};
///@}

/** Maximizer exit code tracking
 */
///@{
/**
 */
static constexpr int NumExitCodes = 10;///< Number of exit codes to track
/** Enumerated exit codes for estimation methods
 * - Error: A Numerical Error was caught.  Did not converge.
 * - Unassigned: Logical error if this is still set
 * - MaxIter: Max iterations exceeded. Did not converge.
 * - MaxBacktracks: Backtracking failed.  Did not converge successfully.
 * - Success: Successful completion
 * - StepSize: Relative Step size was less than epsilon.  Converged successfully.
 * - FunctionValue: Function value change was less than epsilon.  Converged successfully.
 * - GradRatio: Grad ratio was less than epsilon.  Converged successfully.
 * - ModelImprovement: Model predicted improvement is less than epsilon.  Converged Successfully
 * - TrustRegionRadius: Trust region size was less than epsilon.  Converged successfully.
 */
enum class ExitCode : IdxT {TrustRegionRadius = 9,
                            ModelImprovement = 8,
                            GradRatio = 7,
                            FunctionValue = 6,
                            StepSize = 5,
                            Success = 4,
                            MaxBacktracks = 3,
                            MaxIter = 2,
                            Unassigned= 1,
                            Error = 0
                            };
///@}

/** Estimator base class defines the interface for estimator interactions
 * designed to unify the ThreadedEstimator with future GPUEstimator types under a single
 * API.
 *
 * Design notes:
 * Templated on the model type to allow for direct function call for models through
 * the mappel::methods namespace templated model methods.
 */
template<class Model>
class Estimator{
public:
    Estimator(const Model &_model);
    virtual ~Estimator() {}

    virtual std::string name() const =0;
    const Model& get_model();

    /** Maximum likelihood point estimators
     */
    ///@{
    /** Estimate for a single data starting at theta_init, fill in the MLEData struct with the estimated parameter, RLLH, and observed information.
     * Estimation is initialized with theta_init, theta_init is empty, it is estimated with the Heuristic estimator.  If any individual
     * parameters are infinite or are not in the interior of the feasible region, they will be estimated with the Heuristic method.  Valid
     * parameters of theta_init will not be modified in the initialization process.
     *
     * The stencil at the MLE is also returned but can be ignored if
     * not needed as it is available at no extra cost.
     * @param[in] data Model data to estimate for
     * @param[in] theta_init [Optional] Initial theta value.
     * @param[out] mle_data MLEData recording the maximum likelihood estimate and relevant data.
     * @param[out] stencil [Optional] StencilT at the MLE value.
     */
    void estimate_max(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle_data, StencilT<Model> &mle_stencil);
    void estimate_max(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle_data);
    void estimate_max(const ModelDataT<Model> &data, MLEData &mle_data);

   /** Debug estimation for a single data starting at theta_init, fill in the MLEDebugData struct with data including the sequence of evaluated points.
     * Estimation is initialized with theta_init, theta_init is empty, it is estimated with the Heuristic estimator.  If any individual
     * parameters are infinite or are not in the interior of the feasible region, they will be estimated with the Heuristic method.  Valid
     * parameters of theta_init will not be modified in the initialization process.
     *
     * The sequence and sequence_rllh parameters of the MLEDebugData struct record the entire sequence of evaluated points including theta_init and theta_mle,
     * which should be first and last respectively.
     *
     * The stencil at the MLE is also returned but can be ignored if not needed as it is available at no extra cost.
     * @param[in] data Model data to estimate for
     * @param[in] theta_init Initial theta value.
     * @param[out] mle_data MLEDebugData recording the maximum likelihood estimate and relevant data.
     * @param[out] stencil [Optional] StencilT at the MLE value.
     */
     void estimate_max_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEDebugData &mle_data, StencilT<Model> &mle_stencil);
     void estimate_max_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEDebugData &mle_data);

    /** Estimate for a stack of data and fill in the MLEDataStack struct with the estimated parameter, RLLH, and observed information for each data in parallel.
     * Estimation is initialized with theta_init, theta_init is empty, it is estimated with the Heuristic estimator.  If any individual
     * parameters are infinite or are not in the interior of the feasible region, they will be estimated with the Heuristic method.  Valid
     * parameters of theta_init will not be modified in the initialization process.
     * @param[in] data Model data to estimate for
     * @param[in] theta_init [optional] Initial theta value for each image.
     * @param[out] mle MLEStackData records the maximum likelihood estimate, RLLH, and Observed information for each data
     */
    virtual void estimate_max_stack(const ModelDataStackT<Model> &data_stack, const ParamVecT<Model> &theta_init_stack, MLEDataStack &mle_data_stack) = 0;
    void estimate_max_stack(const ModelDataStackT<Model> &data_stack, MLEDataStack &mle_data_stack);
    ///@}


    /** Profile likelihood estimation methods
     */
    ///@{
    double estimate_profile_max(const ModelDataT<Model> &data, const IdxVecT &fixed_idxs,
                                const ParamT<Model> &fixed_theta_init, StencilT<Model> &theta_max);
    virtual void estimate_profile_max(const ModelDataT<Model> &data, const ParamVecT<Model> &fixed_theta_init, ProfileLikelihoodData &profile) = 0;
    ///@}


    /** Profile likelihood bounds computations with VM algorithm
     */
    ///@{
    void estimate_profile_bounds(const ModelDataT<Model> &data, ProfileBoundsData &bounds_est);
    virtual void estimate_profile_bounds_parallel(const ModelDataT<Model> &data, ProfileBoundsData &bounds_est) = 0;
    void estimate_profile_bounds_debug(const ModelDataT<Model> &data, ProfileBoundsDebugData &bounds_est);
    virtual void estimate_profile_bounds_stack(const ModelDataStackT<Model> &data_stack, ProfileBoundsDataStack &bounds_est) = 0;
    ///@}

    /** Run statistics. */
    ///@{
    virtual StatsT get_stats();
    virtual StatsT get_debug_stats()=0;
    virtual void clear_stats();
    IdxVecT get_exit_counts() const { return exit_counts; }
    ///@}

    /* I/O */
    template<class T>
    friend std::ostream& operator<<(std::ostream &out, Estimator<T> &estimator);

protected:
    virtual void compute_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
                                  MLEData &mle_data, StencilT<Model> &mle_stencil)=0;
    virtual void compute_estimate_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
                                        MLEDebugData &mle_data, StencilT<Model> &mle_stencil);
    virtual double compute_profile_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
                                            const IdxVecT &fixed_idxs, StencilT<Model> &max_stencil);
    virtual void compute_profile_bound(const ModelDataT<Model> &data, ProfileBoundsData &est, const VecT &init_step, IdxT param_idx, IdxT which_bound);
    virtual void compute_profile_bound_debug(const ModelDataT<Model> &data, ProfileBoundsDebugData &est);

    void record_walltime(ClockT::time_point start_walltime, int num_estimations);
    virtual void record_exit_code(ExitCode code)=0;

    const Model &model;

    /* statistics */
    int num_estimations = 0;
    double total_walltime = 0.;
    IdxVecT exit_counts; //Vector
};

/**
 * We avoid combining Estimator and ThreadedEstimator classes so that a future GPU implementation can
 * inherit directly from Estimator as it will present a different method for estimate_stack pure virtual
 * member function.  For now all other (CPU) estimators inherit from ThreadedEstimator.
 *
 */
template<class Model>
class ThreadedEstimator : public Estimator<Model> {
public:
    using Estimator<Model>::model;
    ThreadedEstimator(const Model &model);

    void estimate_max_stack(const ModelDataStackT<Model> &data, const ParamVecT<Model> &theta_init_stack, MLEDataStack &mle_data_stack) override;
    void estimate_profile_max(const ModelDataT<Model> &data, const ParamVecT<Model> &theta_init, ProfileLikelihoodData &profile) override;
    void estimate_profile_bounds_parallel(const ModelDataT<Model> &data, ProfileBoundsData &bounds_est) override;
    void estimate_profile_bounds_stack(const ModelDataStackT<Model> &data, ProfileBoundsDataStack &bounds_est_stack) override;

    StatsT get_stats();
    StatsT get_debug_stats();
    void clear_stats();

protected:
    int max_threads;
    int num_threads;
    std::mutex mtx;

    void record_exit_code(ExitCode code) override;
};

template<class Model>
class HeuristicEstimator : public ThreadedEstimator<Model> {
public:
    using Estimator<Model>::model;
    HeuristicEstimator(const Model &model) : ThreadedEstimator<Model>(model) {}

    StatsT get_stats();
    StatsT get_debug_stats();
    std::string name() const {return "HeuristicEstimator";}

private:
    void compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, MLEData &mle_data, StencilT<Model> &mle_stencil) override;
};

template<class Model>
class CGaussHeuristicEstimator : public ThreadedEstimator<Model> {
public:
    using Estimator<Model>::model;
    CGaussHeuristicEstimator(const Model &model) : ThreadedEstimator<Model>(model) {}

    StatsT get_stats();
    StatsT get_debug_stats();
    std::string name() const {return "CGaussHeuristicEstimator";}

private:
    void compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, MLEData &mle_data, StencilT<Model> &mle_stencil) override;
};


template<class Model>
class CGaussMLE : public ThreadedEstimator<Model> {
public:
    using Estimator<Model>::model;
    static const int DefaultIterations;

    CGaussMLE(const Model &model, int num_iterations=DefaultIterations)
        : ThreadedEstimator<Model>(model), num_iterations(num_iterations) {}

    StatsT get_stats();
    StatsT get_debug_stats();
    std::string name() const {return "CGaussMLE";}

private:
    int num_iterations;

    void compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
                          MLEData &mle_data, StencilT<Model> &mle_stencil) override;
    void compute_estimate_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
                                MLEDebugData &mle_data, StencilT<Model> &mle_stencil) override;
};


template<class Model>
class SimulatedAnnealingMaximizer : public ThreadedEstimator<Model> {
public:    
    using Estimator<Model>::model;

    static const int DefaultNumIterations; ///< Default number of SA iterations.
    static const double Default_T_Init; ///< Default SA initial temperature
    static const double DefaultCoolingRate; ///< Default SA cooling rate

    SimulatedAnnealingMaximizer(const Model &model,int num_iterations_=DefaultNumIterations, double T_init_=Default_T_Init, double cooling_rate_=DefaultCoolingRate)
        : ThreadedEstimator<Model>(model),
          num_iterations(num_iterations_),
          T_init(T_init_),
          cooling_rate(cooling_rate_)
    { }

    StatsT get_stats();
    StatsT get_debug_stats();
    std::string name() const {return "SimulatedAnnealingMaximizer";}

private:
    using ThreadedEstimator<Model>::mtx;

    void compute_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle_data, StencilT<Model> &mle_stencil) override;
    void compute_estimate_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEDebugData &mle_data, StencilT<Model> &mle_stencil) override;
    double compute_profile_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const IdxVecT &fixed_idxs, StencilT<Model> &theta_max) override;

    StencilT<Model> anneal(const ModelDataT<Model> &im, const StencilT<Model> &theta_init, const IdxVecT &fixed_params_mask,
                           double &rllh, MatT &sequence, VecT &sequence_rllh);

    int num_iterations; ///< Number of annealing iterations
    double T_init; ///< Initial temperature for determining acceptance ratio
    double cooling_rate; ///< Cooling rate >1.0

    int total_iterations = 0;
    int total_fun_evals = 0;
    int total_der_evals = 0;
    void record_run_statistics(int num_iters, int num_fun_evals, int num_der_evals);
};

template<class Model>
class IterativeMaximizer : public ThreadedEstimator<Model> {
public:
    using Estimator<Model>::model;

    static const int DefaultIterations;

    IterativeMaximizer(const Model &model, int max_iterations=DefaultIterations);

    /* Statistics */
    double mean_iterations();
    double mean_backtracks();
    double mean_fun_evals();
    double mean_der_evals();
    StatsT get_stats();
    StatsT get_debug_stats();
    void clear_stats();
    int get_total_iterations() const { return total_iterations; }
    int get_total_backtracks() const { return total_backtracks; }
    int get_total_fun_evals() const { return total_fun_evals; }
    int get_total_der_evals() const { return total_der_evals; }
    
    /** @brief Perform a local maximization to finish off a simulated annealing run */
    void local_maximize(const ModelDataT<Model> &im, StencilT<Model> &stencil, MLEData &data); //This is used by SimulatedAnnealing to clean up max
    void local_maximize(const ModelDataT<Model> &im, StencilT<Model> &stencil, MLEDebugData &debug_data); //This is used by SimulatedAnnealing to clean up max
    void local_profile_maximize(const ModelDataT<Model> &im, const IdxVecT &fixed_param_idxs, StencilT<Model> &stencil, MLEDebugData &mle); //This is used by SimulatedAnnealing to clean up max

protected:
    using ThreadedEstimator<Model>::mtx;

    /* These parameters control the corrections for indefinite hessian matrices */
    static const double min_eigenvalue_correction_delta; ///<Ensure the minimum eigenvalue is at least this big when correcting indefinite matrix.
    /* These parameters control the adaptive convergence testing */
    static const double convergence_min_function_change_ratio; ///< Convergence criteria: tolerance for function-value change
    static const double convergence_min_step_size_ratio; ///< Convergence criteria: tolerance of relative step size
    /* These parameters control backtracking */
    static const double backtrack_min_ratio; //What is the minimum proportion of the step a backtrack should attempt.
    static const double backtrack_max_ratio; //What is the minimum proportion of the step a backtrack should attempt.
    static const double backtrack_min_linear_step_ratio; //How much improvement in f-val we expect compared to linear in order to not backtrack.
    static const int max_backtracks; //Max # of evaluations to do when backtracking
    /* Parameters controlling profile-bounds solutions with the Venzon&Moolgavkar algorithm */
    static const double min_profile_bound_residual; ///< Minimum residual in quadratic solutions of equation (8) to accept.  Revert to newton step.

    int max_iterations;

    /* Statistics: need to be private so they can be mutex protected */
    int total_iterations = 0;
    int total_backtracks = 0;
    int total_fun_evals = 0;
    int total_der_evals = 0;

    /* Debug Statistics: */
    IdxVecT last_backtrack_idxs;///< Debugging: Stores last set of backtrack_idxs when data.save_seq==true

    class MaximizerData {
    public:
        const ModelDataT<Model> &im;
        ParamT<Model> grad;
        ParamT<Model> step;
        double rllh;

        int nBacktracks=0;
        int nIterations=0;

        IdxVecT fixed_idxs, free_idxs;

        MaximizerData(const Model &model, const ModelDataT<Model> &im, const StencilT<Model> &s, bool save_seq=false);
        MaximizerData(const Model &model, const ModelDataT<Model> &im, const StencilT<Model> &s, double rllh, bool save_seq=false);

        void record_iteration() {record_iteration(theta());}
        /** @brief Record an iteration point (derivatives computed) */
        void record_iteration(const ParamT<Model> &accepted_theta);
        /** @brief Record a backtracked point (no derivative computations performed) Using the saved theta as the default. */
        void record_backtrack(double rejected_rllh) {record_backtrack(theta(), rejected_rllh);}
        /** @brief Record a backtracked point (no derivative computations performed) */
        void record_backtrack(const ParamT<Model> &rejected_theta, double rejected_rllh);
        
        /** @brief Return the saved theta sequence */
        bool has_theta_sequence() const { return max_seq_len>0; }
        IdxT get_sequence_len() const { return seq_len; }
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
        const StencilT<Model>& saved_stencil() const {return current_stencil ? s1 : s0;}
        /** @brief Get the current stencil's theta  */
        const ParamT<Model>& theta() const {return current_stencil ? s0.theta : s1.theta;}
        /** @brief Get the saved stencil's theta  */
        const ParamT<Model>& saved_theta() const {return current_stencil ? s1.theta : s0.theta;}

        void set_fixed_parameters(const IdxVecT &fixed_parameters_idxs);
        bool has_fixed_parameters() const { return !fixed_idxs.is_empty(); }
        IdxT num_fixed_parameters() const { return fixed_idxs.n_elem; }
    protected:
        static const int DefaultMaxSeqLength; ///< Default maximum length of sequence to perpare to save if debugging.
        const IdxT num_params;
        StencilT<Model> s0,s1; //These two stencils will be alternated as the current and old stencil points
        bool current_stencil; //This alternates to indicated weather s0 or s1 is the current stencil

        int max_seq_len=0;
        int seq_len=0;
        ParamVecT<Model> theta_seq;
        VecT seq_rllh;
        IdxVecT backtrack_idxs;

        void expand_max_seq_len();
    };

    void record_run_statistics(const MaximizerData &data);


    void compute_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle_data, StencilT<Model> &mle_stencil) override;
    void compute_estimate_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEDebugData &mle_data, StencilT<Model> &mle_stencil) override;
    double compute_profile_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const IdxVecT &fixed_idxs, StencilT<Model> &theta_max) override;
    void compute_profile_bound(const ModelDataT<Model> &data, ProfileBoundsData &est, const VecT &init_step, IdxT param_idx, IdxT which_bound) override;
    void compute_profile_bound_debug(const ModelDataT<Model> &data, ProfileBoundsDebugData &bounds) override;


    bool backtrack(MaximizerData &data);
    bool profile_bound_backtrack(MaximizerData &data, IdxT fixed_idx, double target_rllh,double old_fval, const VecT& fgrad);

    virtual void maximize(MaximizerData &data)=0;
    virtual void solve_profile_bound(MaximizerData &data, MLEData &mle, double llh_delta, IdxT fixed_idx, IdxT which_bound);

    bool convergence_test_grad_ratio(const VecT &grad, double fval);
    bool convergence_test_step_size(const VecT &new_theta, const VecT &old_theta);
};


template<class Model>
class NewtonDiagonalMaximizer : public IterativeMaximizer<Model> {
public:
    using Estimator<Model>::model;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    NewtonDiagonalMaximizer(const Model &model, int max_iterations=IterativeMaximizer<Model>::DefaultIterations)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonDiagonalMaximizer";}

private:
    void maximize(MaximizerData &data) override;
};

template<class Model>
class NewtonMaximizer : public IterativeMaximizer<Model> {
public:
    using Estimator<Model>::model;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    NewtonMaximizer(const Model &model, int max_iterations=IterativeMaximizer<Model>::DefaultIterations)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "NewtonMaximizer";}

private:
    void maximize(MaximizerData &data) override;

    void solve_profile_bound(MaximizerData &data, MLEData &mle, double llh_delta, IdxT fixed_idx, IdxT which_bound) override;
};

template<class Model>
class QuasiNewtonMaximizer : public IterativeMaximizer<Model> {
public:
    using Estimator<Model>::model;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    QuasiNewtonMaximizer(const Model &model, int max_iterations=IterativeMaximizer<Model>::DefaultIterations)
        : IterativeMaximizer<Model>(model,max_iterations) {}

    inline std::string name() const {return "QuasiNewtonMaximizer";}

private:
    void maximize(MaximizerData &data) override;
};

template<class Model>
class TrustRegionMaximizer : public IterativeMaximizer<Model> {
public:
    using Estimator<Model>::model;
    using MaximizerData = typename IterativeMaximizer<Model>::MaximizerData;

    static const double rho_cauchy_min;///<Minimum acceptable rho for cauchy point: Coleman beta / Bellavia beta_1
    static const double rho_obj_min;///<Minimum acceptable rho: Coleman mu / Bellavia beta_2
    static const double rho_obj_opt;///<Optimal step rho: Coleman eta / Bellavia beta_2
    static const double trust_radius_decrease_min;///< Smallest alowable trust radius decrease ratio: Coleman gamma_0 / Bellavia alpha_1
    static const double trust_radius_decrease;///< Trust radius decrease ratio to step size: Coleman gamma_1 / Bellavia alpha_2
    static const double trust_radius_increase;///< Trust radius increase ratio: Coleman gamma_2 / Bellavia alpha_3
    static const double convergence_min_trust_radius; ///< Convergence criteria: Minimum trust region radius

    TrustRegionMaximizer(const Model &model, int max_iterations=IterativeMaximizer<Model>::DefaultIterations)
        : IterativeMaximizer<Model>(model,max_iterations) {}
    
    inline std::string name() const {return "TrustRegionMaximizer";}
private:
    void maximize(MaximizerData &data) override;
};

} /* namespace mappel::estimator */

} /* namespace mappel */

#endif /* MAPPEL_ESTIMATOR_H */
