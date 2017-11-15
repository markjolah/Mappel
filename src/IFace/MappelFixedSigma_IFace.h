/** @file Ma.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for Mappel2D_IFace.
 */

#ifndef _MAPPEL2D_IFACE
#define _MAPPEL2D_IFACE

#include "Mappel_IFace.h"

namespace mappel {

template<class Model>
class Mappel2D_IFace : public Mappel_IFace<Model>{
// public:
//     Mappel2D_IFace(std::string name) : Mappel_IFace<Model>(name) {};

protected:
    using Mappel_IFace<Model>::obj;

    typename Model::ImageT getImage();
    typename Model::ImageStackT getImageStack();
};


template<class Model>
typename Model::ImageT Mappel2D_IFace<Model>::getImage()
{
    return this->getDMat();
}

template<class Model>
typename Model::ImageStackT Mappel2D_IFace<Model>::getImageStack()
{
    return this->getDStack();
}

} /* namespace mappel */

#endif /*_MAPPEL2D_IFACE */
