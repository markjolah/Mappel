#ifndef WIN32
    #include <sched.h>
#endif
#include <cctype>
#include <omp.h>
#include "Mappel/util.h"

namespace mappel {

void enable_all_cpus() {
    #ifndef WIN32
    /* This seems necessary on linux because openblas fucks with it ARGH! */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for(int i=0;i<omp_get_max_threads();i++) CPU_SET(i, &cpuset);
    sched_setaffinity(0,sizeof(cpuset),&cpuset);
    #endif
}


bool istarts_with(const char* s, const char* pattern)
{
    while(*s && *pattern) {
        if(toupper(*s++) != toupper(*pattern++)) return false;
    }
    return !*pattern;  //True if the pattern is over
}

bool istarts_with(const std::string& str, const char* pattern)
{
    auto s = str.begin();
    while(*s && *pattern) {
        if(toupper(*s++) != toupper(*pattern++)) return false;
    }
    return !*pattern;  //True if the pattern is over
}


const char * icontains(const char* s, const char* pattern)
{
    const char *p=pattern;
    const char *start=nullptr;
    while(*s) {
        if(toupper(*s++) == toupper(*p)) { //found match
            if(p==pattern) start=p;
            if (!*++p) return start; //matched whole string
        } else {
            p=pattern;
        }
    }
    return nullptr;
}


int maxidx(const VecT &v)
{
    double maxval=-INFINITY;
    int maxn=-1;
    for(unsigned n=0; n<v.n_elem; n++) {
        if(maxval<v(n)){
            maxn=n;
            maxval=v(n);
        }
    }
    return maxn;
}

std::ostream& operator<<(std::ostream &out,const StatsT &stats)
{
    for(auto& stat: stats) out<<"\t"<<stat.first<<" = "<<stat.second<<"\n";
    return out;
}


} /* namespace mappel */
