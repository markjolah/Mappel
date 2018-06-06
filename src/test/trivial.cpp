#include "Mappel/Gauss1DMAP.h"
#include <iostream>
#include <type_traits>
using namespace mappel;
using namespace std;


int main(){
    cout<<"is_trivially_copyable<Gauss1DMAP>() ->"<<is_trivially_copyable<Gauss1DMAP>()<<endl;
    cout<<"is_trivially_copyable<Gauss1DModel>() ->"<<is_trivially_copyable<Gauss1DModel>()<<endl;
    cout<<"is_trivially_copyable<ImageFormat1DBase>() ->"<<is_trivially_copyable<ImageFormat1DBase>()<<endl;
    cout<<"is_trivially_copyable<ParallelRngManagerT>() ->"<<is_trivially_copyable<ParallelRngManagerT>()<<endl;
    cout<<"is_trivially_copyable<PointEmitterModel>() ->"<<is_trivially_copyable<PointEmitterModel>()<<endl;
    cout<<"is_trivially_copyable<CompositeDist>() ->"<<is_trivially_copyable<CompositeDist>()<<endl;
    cout<<"is_trivially_copyable<PoissonNoise1DObjective>() ->"<<is_trivially_copyable<PoissonNoise1DObjective>()<<endl;
    cout<<"is_trivially_copyable<MAPEstimator>() ->"<<is_trivially_copyable<MAPEstimator>()<<endl;
    cout<<"is_trivially_copyable<vector<double>>() ->"<<is_trivially_copyable<vector<double>>()<<endl;
    cout<<"is_trivially_copyable<std::string>() ->"<<is_trivially_copyable<std::string>()<<endl;
    
    return 0;
}
