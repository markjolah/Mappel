
#include "rng.h"

#include <ctime>
#include <omp.h>

namespace mappel {

uint64_t  make_seed()
{
    uint32_t lo,hi;
    uint64_t seed;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    seed=( ((uint64_t)lo) << 32) | hi;
    uint64_t walltime=time(NULL);
    seed^=walltime<<32 | (0xFFFFFFFF & walltime);
//     std::cout<<"Generated seed: "<<seed<<std::endl;
    return seed;
}

RNG make_parallel_rng_stream(uint64_t seed)
{
    RNG rng(seed);
    int size=omp_get_num_threads();
    int rank=omp_get_thread_num();
    rng.split(size,rank);
    return rng;
}

} /* namespace mappel */
