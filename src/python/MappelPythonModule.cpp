/** @file MappelPythonModule.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2018
 * @brief The instantiation of the mappel python module
 *
 * This is the core for the mappel::python interface
 */
#include "py11_armadillo_iface.h"
#include "py11_mappel_iface.h"

#include "Gauss1DMLE.h"


// namespace py = pybind11;
// namespace py11_armadillo;



// NDArrayDoubleT* test(NDArrayDoubleT array)
// {
//     
//     
//     auto *a2 = new NDArrayDoubleT(array);
//     std::cout<<"Ndim:"<<a2->ndim()<<std::endl;
//     std::cout<<"itemsize:"<<a2->itemsize()<<std::endl;
//     std::cout<<"size:"<<a2->size()<<std::endl;
//     arma::vec c = viewAsArmaCol<double>(array);
//     std::cout<<"c:"<<c.t()<<std::endl;
//     std::cout<<"c.n_elem:"<<c.n_elem<<std::endl;
//     return a2;
// }


PYBIND11_MODULE(mappel, M)
{
    M.doc()="Mappel Python Interface!";
    mappel::python::bindMappelModel<mappel::Gauss1DMLE>(M);
}

