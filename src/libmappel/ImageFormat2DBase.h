/** @file ImageFormat2DBase.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for ImageFormat2DBase.
 *
 * The virtual base class for all point 2D image based emitter Models and Objectives
 */

#ifndef _IMAGEFORMAT2DBASE_H
#define _IMAGEFORMAT2DBASE_H

#include "util.h"
#include <sstream>

namespace mappel {

/** @brief A virtual base class for 2D image localization objectives
 *
 * This class should be inherited virtually by both the model and the objective so that
 * the common image information and functions are available in both Model and Objective classes hierarchies
 * 
 */
class ImageFormat2DBase {
public:
    using ImageT = arma::mat; /**< A type to represent image data*/
    using ImageStackT = arma::cube; /**< A type to represent image data stacks */

    static const int constexpr num_dim=2;
    static const int constexpr min_size=3; /**< Minimum size along any dimension of the image.  Prevents "too small to be meaningfull" images. */
    /* Model parameters */
    const IVecT size; /**< The number of pixels in the X and Y directions, given as [X,Y].  Note that images have shape [size(1),size(0)], Y is rows X is columns.   */

    ImageFormat2DBase(const IVecT &size);
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(int n) const;
    int size_image_stack(const ImageStackT &stack) const;
    ImageT get_image_from_stack(const ImageStackT &stack, int n) const;
};

/* Inline Method Definitions */

inline
ImageFormat2DBase::ImageT
ImageFormat2DBase::make_image() const
{
    return ImageT(size(1),size(0)); //Images are size [Y X]
}

inline
ImageFormat2DBase::ImageStackT
ImageFormat2DBase::make_image_stack(int n) const
{
    return ImageStackT(size(1),size(0),n);
}

inline
int ImageFormat2DBase::size_image_stack(const ImageStackT &stack) const
{
    return static_cast<int>(stack.n_slices);
}


inline
ImageFormat2DBase::ImageT
ImageFormat2DBase::get_image_from_stack(const ImageStackT &stack,int n) const
{
    return stack.slice(n);
}

/* Templated Function Definitions */

template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat2DBase,Model>::value,typename Model::ImageT>::type
model_image(const Model &model, const typename Model::Stencil &s)
{
    auto im=model.make_image();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=xposition=column; j=yposition=row
        im(j,i)=model.pixel_model_value(i,j,s);
        if( im(j,i) <= 0.){
            //Model value must be positive for grad to be defined
            std::ostringstream os;
            os<<"Non positive model value encountered: "<<im(j,i)<<" at j,i=("<<j<<","<<i<<")";
            throw MappelException("model_image",os.str());
        }
    }
    return im;
}

/** @brief  */
template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value,typename Model::MatT>::type
fisher_information(const Model &model, const typename Model::Stencil &s)
{
    auto fisherI=model.make_param_mat();
    fisherI.zeros();
    auto pgrad=model.make_param();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        double model_val=model.pixel_model_value(i,j,s);
        model.pixel_grad(i,j,s,pgrad);
        for(int c=0; c<model.num_params; c++) for(int r=0; r<=c; r++) {
            fisherI(r,c) += pgrad(r)*pgrad(c)/model_val; //Fill upper triangle
        }
    }
    model.prior_hess_update(s.theta,fisherI); /* As appropriate for MAP/MLE: Add diagonal hession of log of prior for params theta */
    return fisherI;
}



} /* namespace mappel */

#endif /* _IMAGEFORMAT2DBASE_H */
