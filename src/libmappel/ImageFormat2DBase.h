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
protected:
    ImageFormat2DBase()=default; //For virtual inheritance simplification
public:
    using ImageCoordT = uint32_t; /**< Image size coordinate storage type  */
    using ImagePixelT = double; /**< Image pixel storage type */
    
    template<class CoordT> using ImageSizeShapeT = arma::Col<CoordT>;  /**<Shape of the data type to store a single image's coordinates */
    template<class CoordT> using ImageSizeVecShapeT = arma::Mat<CoordT>; /**<Shape of the data type to store a vector of image's coordinates */    
    using ImageSizeT = ImageSizeShapeT<ImageCoordT>; /**< Data type for a single image size */ 
    using ImageSizeVecT = ImageSizeVecShapeT<ImageCoordT>; /**< Data type for a sequence of image sizes */

    template<class PixelT> using ImageShapeT = arma::Mat<PiexlT>; /**< Shape of the data type for a single image */
    template<class PixelT> using ImageStackShapeT = arma::Cube<PixelT>; /**< Shape of the data type for a sequence of images */
    using ImageT = ImageShapeT<ImagePixelT>; /**< Data type to represent single image*/
    using ImageStackT = ImageStackShapeT<ImagePixelT>; /**< Data type to represent a sequence of images */

    constexpr static ImageCoordT num_dim = 1;  /**< Number of image dimensions. */
    constexpr static ImageCoordT min_size = 3; /**< Minimum size along any dimension of the image. */

    /* Model parameters */
    ImageSizeT size; /**< Number of pixels as [sizeY(#rows), sizeX(#cols)] Dimensions */
    ImageCoordT num_pixels; /**< Total number of pixels in image */

    ImageFormat2DBase(ImageSizeT size_);
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(ImageCoordT n) const;
    ImageCoordT get_size_image_stack(const ImageStackT &stack) const;
    ImageT get_image_from_stack(const ImageStackT &stack, ImageCoordT n) const;
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
ImageFormat1DBase::ImageCoordT
ImageFormat2DBase::size_image_stack(const ImageStackT &stack) const
{
    return static_cast<ImageSizeT>(stack.n_slices);
}


inline
ImageFormat2DBase::ImageT
ImageFormat2DBase::get_image_from_stack(const ImageStackT &stack,ImageCoordT n) const
{
    return stack.slice(n);
}

/* Templated Function Definitions */

template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat2DBase,Model>::value,typename Model::ImageT>::type
model_image(const Model &model, const typename Model::Stencil &s)
{
    auto im=model.make_image();
    for(ImageCoordT i=0;i<model.size(0);i++) for(ImageCoordT j=0;j<model.size(1);j++) {  // i=xposition=column; j=yposition=row
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
    return fisherI;
}



} /* namespace mappel */

#endif /* _IMAGEFORMAT2DBASE_H */
