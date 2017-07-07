/** @file Mappel2D_Iface.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for Mappel2D_Iface.
 */

#ifndef _MAPPEL2D_IFACE
#define _MAPPEL2D_IFACE

#include "Mappel_IFace.h"

namespace mappel {

template<class Model>
class Mappel2D_Iface : public Mappel_Iface<Model>{
public:
    Mappel2D_Iface(std::string name) : Mappel_Iface<Model>(name) {};

protected:
    using Mappel_Iface<Model>::obj;

    typename Model::ImageT getImage();
    typename Model::ImageStackT getImageStack();
    typename Model::ImageStackT makeImageStack(int count);

    void objConstruct();
};

template<class Model>
void Mappel2D_Iface<Model>::objConstruct()
{
    this->checkNumArgs(1,2);
    Model *model=new Model(this->getIVec(),this->getDVec());
    this->outputMXArray(Handle<Model>::makeHandle(model));
}

template<class Model>
typename Model::ImageT Mappel2D_Iface<Model>::getImage()
{
    return this->getDMat();
}


template<class Model>
typename Model::ImageStackT Mappel2D_Iface<Model>::getImageStack()
{
    return this->getDStack();
}

template<class Model>
typename Model::ImageStackT Mappel2D_Iface<Model>::makeImageStack(int count)
{
    return this->makeDStack(obj->size(1), obj->size(0), count);
}

} /* namespace mappel */

#endif /*_MAPPEL2D_IFACE */
