
/** @file OMPExceptionCatcher.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2019
 * @copyright See LICENSE file
 * @brief A lightweight class for managing C++ exception handling strategies for OpenMP methods.
 *
 * OpenMP code must catch any exceptions that may have been thrown before exiting the OpenMP block.
 * This class acts as lightweight wrapper that allows an arbitrary function or lambda expression to be run
 * safely and efficiently in OMP even if it might throw exceptions.  We employ one of 4 possible strategies
 * as determined By the OMPExceptionCatcher::Strategies enum.
 *
 * Strategy's :
 * OMPExceptionCatcher::Strategies::DoNotTry -- Don't even try,  this is a null op to completely disable
 *                                              this class's effect.
 * OMPExceptionCatcher::Strategies::Continue -- Catch exceptions and keep going
 * OMPExceptionCatcher::Strategies::Abort    -- Catch exceptions and abort
 * OMPExceptionCatcher::Strategies::RethrowFirst  -- Re-throws first exception thrown by any thread
 *
 *
 * Example useage:
 * OMPExceptionCatcher catcher(OMPExceptionCatcher<>::Strategies::Continue);
 * #pragma omp parallel for
 * for(int n=0; n < N; n++) catcher.run([&]{ my_ouput(n)=do_my calulations(args(n)); }
 * catcher.rethrow(); //Required only if you ever might use RethrowFirst strategy
 */

#ifndef OMP_EXCEPTION_CATCHER_H
#define OMP_EXCEPTION_CATCHER_H

#include<exception>
#include<mutex>
#include<functional>
#include<cstdint>

namespace omp_exception_catcher {

namespace impl_ {
//IntType is a dummy just to allow everything to be a template and static member initialization
//to be defined in a header-only file
template<class IntType=uint32_t>
class OMPExceptionCatcher
{
public:
    enum class Strategy:IntType {DoNotTry, Continue, Abort, RethrowFirst};
private:
    static Strategy GlobalDefaultStrategy;
public:
    static void setGlobalDefaultStrategy(Strategy s) { GlobalDefaultStrategy = s; }
    OMPExceptionCatcher(): ex(nullptr), strategy(GlobalDefaultStrategy) {}

    OMPExceptionCatcher(Strategy strategy_): ex(nullptr), strategy(strategy_) {}

    void rethrow() const { if(strategy==Strategy::RethrowFirst && ex) std::rethrow_exception(ex); }

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
typename OMPExceptionCatcher<IntType>::Strategy
OMPExceptionCatcher<IntType>::GlobalDefaultStrategy = OMPExceptionCatcher<IntType>::Strategy::RethrowFirst;

} /* namespace omp_exception_catcher::impl_ */

using OMPExceptionCatcher = impl_::OMPExceptionCatcher<uint32_t>;

} /* namespace omp_exception_catcher */

#endif
