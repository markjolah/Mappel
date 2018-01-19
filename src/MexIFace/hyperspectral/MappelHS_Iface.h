/** @file MappelHS_Iface.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for MappelHS_Iface.
 */

#ifndef _MAPPELHS_IFACE
#define _MAPPELHS_IFACE

#include "Mappel_Iface.h"

template<class Model>
class MappelHS_Iface : public Mappel_Iface<Model>{
public:
    MappelHS_Iface(std::string name) : Mappel_Iface<Model>(name) {};

protected:
    using Mappel_Iface<Model>::obj;

    typename Model::ImageT getImage();
    typename Model::ImageStackT getImageStack();
    typename Model::ImageStackT makeImageStack(int count);

    void objConstruct();
};

template<class Model>
void MappelHS_Iface<Model>::objConstruct()
{
    this->checkNumArgs(1,2);
    auto size=this->getIVec();
    auto sigma=this->getDVec();
    Model *model=new Model(size,sigma);
    this->outputMXArray(Handle<Model>::makeHandle(model));
}


template<class Model>
typename Model::ImageT MappelHS_Iface<Model>::getImage()
{
    return this->getDStack();
}


template<class Model>
typename Model::ImageStackT MappelHS_Iface<Model>::getImageStack()
{
    return this->getDHyperStack();
}


template<class Model>
typename Model::ImageStackT MappelHS_Iface<Model>::makeImageStack(int count) 
{
    return this->makeDHyperStack(obj->size(2), obj->size(1), obj->size(0), count);
}


#endif /*_MAPPELHS_IFACE */
