/** @file ImageFormat2DBase.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class declaration and inline and templated functions for ImageFormat2DBase.
 *
 * The virtual base class for all point 2D image based emitter Models and Objectives
 */

#ifndef _MAPPEL_IMAGEFORMAT2DBASE_H
#define _MAPPEL_IMAGEFORMAT2DBASE_H

#include "util.h"

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

    template<class PixelT> using ImageShapeT = arma::Mat<PixelT>; /**< Shape of the data type for a single image */
    template<class PixelT> using ImageStackShapeT = arma::Cube<PixelT>; /**< Shape of the data type for a sequence of images */
    using ImageT = ImageShapeT<ImagePixelT>; /**< Data type to represent single image*/
    using ImageStackT = ImageStackShapeT<ImagePixelT>; /**< Data type to represent a sequence of images */

    constexpr static ImageCoordT num_dim = 2;  /**< Number of image dimensions. */
    constexpr static ImageCoordT global_min_size = 3; /**< Minimum size along any dimension of the image. */
    constexpr static ImageCoordT global_max_size = 4096; /**< Maximum size along any dimension of the image.  This is insanely big to catch obvious errors */

    ImageFormat2DBase(const ImageSizeT &size);
    
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(ImageCoordT n) const;
    ImageCoordT get_size_image_stack(const ImageStackT &stack) const;
    ImageT get_image_from_stack(const ImageStackT &stack, ImageCoordT n) const;

    template<class ImT>
    void set_image_in_stack(ImageStackT &stack, ImageCoordT n, const ImT &im) const;
    
    ImageSizeT get_size() const;
    ImageCoordT get_size(IdxT idx) const;
    ImageCoordT get_num_pixels() const;
    void set_size(const ImageSizeT &size_);
    void check_image_shape(const ImageT &im) const;
    void check_image_shape(const ImageStackT &ims) const;

protected:
    ImageSizeT size; /**< Number of pixels in X dimension for 1D image */

    static void check_size(const ImageSizeT &size_);
};

/* Inline Method Definitions */

inline
ImageFormat2DBase::ImageSizeT  
ImageFormat2DBase::get_size() const
{ return size; }

inline
ImageFormat2DBase::ImageCoordT 
ImageFormat2DBase::get_num_pixels() const
{ return size(0)*size(1); }


inline
ImageFormat2DBase::ImageT
ImageFormat2DBase::make_image() const
{
    return ImageT(size(1),size(0)); //Images are size [Y X]
}

inline
ImageFormat2DBase::ImageStackT
ImageFormat2DBase::make_image_stack(ImageCoordT n) const
{
    return ImageStackT(size(1),size(0),n);
}

inline
ImageFormat2DBase::ImageCoordT
ImageFormat2DBase::get_size_image_stack(const ImageStackT &stack) const
{
    return static_cast<ImageCoordT>(stack.n_slices);
}

inline
ImageFormat2DBase::ImageT
ImageFormat2DBase::get_image_from_stack(const ImageStackT &stack,ImageCoordT n) const
{
    return stack.slice(n);
}

template<class ImT>
void 
ImageFormat2DBase::set_image_in_stack(ImageStackT &stack, ImageCoordT n, const ImT &im) const
{
    stack.slice(n) = im;
}

namespace methods {

    template<class Model>
    typename std::enable_if<std::is_base_of<ImageFormat2DBase,Model>::value,typename Model::ImageT>::type
    model_image(const Model &model, const typename Model::Stencil &s)
    {
        auto im = model.make_image();
        auto size = model.get_size();
        for(ImageCoordT<Model> i=0;i<size(0);i++) for(ImageCoordT<Model> j=0;j<size(1);j++){//i=Xpos=col; j=Ypos=row
            im(j,i) = model.pixel_model_value(i,j,s);
        }
        return im;
    }
    
} /* namespace mappel::methods */


} /* namespace mappel */

#endif /* _MAPPEL_IMAGEFORMAT2DBASE_H */
