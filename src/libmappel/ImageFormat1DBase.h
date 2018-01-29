/** @file ImageFormat1DBase.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for ImageFormat1DBase.
 *
 * The virtual base class for all point 2D image based emitter Models and Objectives
 */

#ifndef _IMAGEFORMAT1DBASE_H
#define _IMAGEFORMAT1DBASE_H

#include "util.h"

namespace mappel {

/** @brief A virtual base class for 2D image localization objectives
 *
 * This class should be inherited virtually by both the model and the objective so that
 * the common image information and functions are available in both Model and Objective classes hierarchies
 * 
 */
class ImageFormat1DBase {
protected:
    ImageFormat1DBase()=default; //For virtual inheritance simplification
public:
    using ImageCoordT = uint32_t; /**< Image size coordinate storage type  */
    using ImagePixelT = double; /**< Image pixel storage type */
    
    template<class CoordT> using ImageSizeShapeT = CoordT;  /**<Shape of the data type to store 1-image's coordinates */
    template<class CoordT> using ImageSizeVecShapeT = arma::Col<CoordT>; /**<Shape of the data type to store a vector of image's coordinates */    
    using ImageSizeT = ImageSizeShapeT<ImageCoordT>; /**< Data type for a single image size */ 
    using ImageSizeVecT = ImageSizeVecShapeT<ImageCoordT>; /**< Data type for a sequence of image sizes */

    template<class PixelT> using ImageShapeT = arma::Col<PixelT>; /**< Shape of the data type for a single image */
    template<class PixelT> using ImageStackShapeT = arma::Mat<PixelT>; /**< Shape of the data type for a sequence of images */
    using ImageT = ImageShapeT<ImagePixelT>; /**< Data type to represent single image*/
    using ImageStackT = ImageStackShapeT<ImagePixelT>; /**< Data type to represent a sequence of images */

    constexpr static ImageCoordT num_dim = 1;  /**< Number of image dimensions. */
    constexpr static ImageCoordT min_size = 3; /**< Minimum size along any dimension of the image. */
    
    ImageFormat1DBase(ImageSizeT size_);
    ImageFormat1DBase(const arma::Col<ImageSizeT> &size_);
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(ImageCoordT n) const;
    ImageCoordT get_size_image_stack(const ImageStackT &stack) const;
    ImageT get_image_from_stack(const ImageStackT &stack, ImageCoordT n) const;

    template<class ImT>
    void set_image_in_stack(ImageStackT &stack, ImageCoordT n, ImT&&im) const;
    
    ImageSizeT get_size() const;
    ImageCoordT get_num_pixels() const;
    void set_size(const ImageSizeT &size_);
    void check_image_shape(const ImageT &im) const;
    void check_image_shape(const ImageStackT &ims) const;
protected:
    /* Model parameters */
    ImageSizeT size; /**< Number of pixels in X dimension for 1D image */

    static void check_size(const ImageSizeT &size_);
};

/* Inline Method Definitions */
inline
ImageFormat1DBase::ImageSizeT  
ImageFormat1DBase::get_size() const
{ return size; }

inline
ImageFormat1DBase::ImageCoordT 
ImageFormat1DBase::get_num_pixels() const
{ return size; }

inline
ImageFormat1DBase::ImageT
ImageFormat1DBase::make_image() const
{
    return ImageT(size); 
}

inline
ImageFormat1DBase::ImageStackT
ImageFormat1DBase::make_image_stack(ImageCoordT n) const
{
    return ImageStackT(size,n);
}

inline
ImageFormat1DBase::ImageCoordT 
ImageFormat1DBase::get_size_image_stack(const ImageStackT &stack) const
{
    return static_cast<ImageSizeT>(stack.n_cols);
}

inline
ImageFormat1DBase::ImageT
ImageFormat1DBase::get_image_from_stack(const ImageStackT &stack,ImageCoordT n) const
{
    return stack.col(n);
}

template<class ImT>
void ImageFormat1DBase::set_image_in_stack(ImageStackT &stack, ImageCoordT n, ImT&&im  ) const
{
    stack.col(n)=im;
}


/* Templated Methods Definitions */
namespace methods {
    template<class Model, typename = IsSubclassT<Model,ImageFormat1DBase>>
    ImageT<Model>
    model_image(const Model &model, const StencilT<Model> &s)
    {
        auto im = model.make_image();
        for(ImageCoordT<Model> i=0; i<model.get_size(); i++)  im(i) = model.pixel_model_value(i,s);
        return im;
    }
} /* namespace mappel::methods */

} /* namespace mappel */

#endif /* _IMAGEFORMAT1DBASE_H */
