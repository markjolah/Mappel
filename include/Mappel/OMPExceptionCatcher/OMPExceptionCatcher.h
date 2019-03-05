
/** @file OMPExceptionCatcher.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2019
 * @copyright See LICENSE file
 * @brief A lightweight class for managing C++ exception handling strategies for OpenMP methods.
 *
 * OpenMP code must catch any exceptions that may have been thrown before exiting the OpenMP block.
 * This class acts as lightweight wrapper that allows an arbitrary function or lambda expression to be run
 * safely and efficiently in OMP even if it might throw exceptions.  We employ one of 4 possible strategies
 * as determined By the omp_exception_catcher::Strategies enum.
 *
 * Strategy's :
 * omp_exception_catcher::Strategies::DoNotTry -- Don't even try,  this is a null op to completely disable
 *                                              this class's effect.
 * omp_exception_catcher::Strategies::Continue -- Catch exceptions and keep going
 * omp_exception_catcher::Strategies::Abort    -- Catch exceptions and abort
 * omp_exception_catcher::Strategies::RethrowFirst  -- Re-throws first exception thrown by any thread.
 *
 *
 * Example usage:
 * omp_exception_catcher::OMPExceptionCatcher catcher(omp_exception_catcher::Strategies::Continue);
 * #pragma omp parallel for
 * for(int n=0; n < N; n++) catcher.run([&]{ my_output(n)=do_my_calculations(args(n)); })
 * catcher.rethrow(); //Required only if you ever might use RethrowFirst strategy
 */

#ifndef OMP_EXCEPTION_CATCHER_H
#define OMP_EXCEPTION_CATCHER_H

#include <exception>
#include <mutex>
#include <functional>
#include <cstdint>

namespace omp_exception_catcher {

enum class Strategy {DoNotTry, Continue, Abort, RethrowFirst};

namespace impl_ {
/** Implementation of OMPExceptionCatcher
 *
 * Note: The template variable is a dummy.  It exists solely to allow this class to be a template,
 * which makes it header-only and allows static member initialization to be defined in the header file.
 */
template<class _dummy=void>
class OMPExceptionCatcher
{
    static Strategy GlobalDefaultStrategy; //Strategy::RethrowFirst
public:
    static void setGlobalDefaultStrategy(Strategy s) { GlobalDefaultStrategy = s; }

    /** Construct a new OMPExceptionCatcher using the GlobalDefaultStrategy
     */
    OMPExceptionCatcher(): ex(nullptr), strategy(GlobalDefaultStrategy) {}

    /** Construct a new OMPExceptionCatcher using the given strategy
     */
    OMPExceptionCatcher(Strategy strategy_): ex(nullptr), strategy(strategy_) {}

    /** Rethrow any stored exceptions
     * Should only be called from single-threaded blocks of code
     */
    void rethrow() const { if(strategy==Strategy::RethrowFirst && ex) std::rethrow_exception(ex); }

    /** Run a function in parallel code and prevent exceptions escaping.
     *
     * Runs any function with any set of parameters and applies the chosen exception catching Strategy
     * to prevent any exceptions escaping.  This function is thread-safe designed to be called in parallel
     * code blocks.
     * @param[in] func function to call
     * @param[in] params... Possibly empty variadic set of parameters to call.
     *
     */
    template<class Function, class... Parameters>
    void run(Function func, Parameters... params) {
        switch(strategy) {
            case Strategy::DoNotTry:
                func(params...);
                break;
            case Strategy::Continue:
                try { func(params...); }
                catch (...) { }
                break;
            case Strategy::Abort:
                try { func(params...); }
                catch (...) { std::abort(); }
                break;
            case Strategy::RethrowFirst:
                try { func(params...); }
                catch (...) { capture(); }
                break;
        }
    }

private:
    std::exception_ptr ex;
    std::mutex lock;
    Strategy strategy;

    void capture() {
        std::unique_lock<std::mutex> guard(lock);
        if(!ex) ex = std::current_exception();
    }
};

template<class IntType>
Strategy OMPExceptionCatcher<IntType>::GlobalDefaultStrategy = Strategy::RethrowFirst;

} /* namespace omp_exception_catcher::impl_ */

/** A class to run and catch exceptions in parallel code allowing various exception management strategies
 */
using OMPExceptionCatcher = impl_::OMPExceptionCatcher<>;

} /* namespace omp_exception_catcher */

#endif
