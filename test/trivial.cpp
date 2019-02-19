#include "Mappel/Gauss1DMAP.h"
#include <iostream>
#include <type_traits>
using namespace mappel;
using namespace std;

// workaround missing "is_trivially_copyable" in g++ < 5.0
#if __GNUG__ && __GNUC__ < 5
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

int main(){
    cout<<"is_trivially_copyable<Gauss1DMAP>() ->"<<IS_TRIVIALLY_COPYABLE(Gauss1DMAP)<<endl;
    cout<<"is_trivially_copyable<Gauss1DModel>() ->"<<IS_TRIVIALLY_COPYABLE(Gauss1DModel)<<endl;
    cout<<"is_trivially_copyable<ImageFormat1DBase>() ->"<<IS_TRIVIALLY_COPYABLE(ImageFormat1DBase)<<endl;
    cout<<"is_trivially_copyable<ParallelRngManagerT>() ->"<<IS_TRIVIALLY_COPYABLE(ParallelRngManagerT)<<endl;
    cout<<"is_trivially_copyable<PointEmitterModel>() ->"<<IS_TRIVIALLY_COPYABLE(PointEmitterModel)<<endl;
    cout<<"is_trivially_copyable<CompositeDist>() ->"<<IS_TRIVIALLY_COPYABLE(CompositeDist)<<endl;
    cout<<"is_trivially_copyable<PoissonNoise1DObjective>() ->"<<IS_TRIVIALLY_COPYABLE(PoissonNoise1DObjective)<<endl;
    cout<<"is_trivially_copyable<MAPEstimator>() ->"<<IS_TRIVIALLY_COPYABLE(MAPEstimator)<<endl;
    cout<<"is_trivially_copyable<vector<double>>() ->"<<IS_TRIVIALLY_COPYABLE(vector<double>)<<endl;
    cout<<"is_trivially_copyable<std::string>() ->"<<IS_TRIVIALLY_COPYABLE(std::string)<<endl;
    
    return 0;
}
