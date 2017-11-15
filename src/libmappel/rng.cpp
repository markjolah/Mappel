
#include "rng.h"
#include "trng/lcg64_shift.hpp"

namespace mappel {

    parallel_rng::ParallelRngManager<ParallelRngT> rng_manager; //Default RNG manager

} /* namespace mappel */
