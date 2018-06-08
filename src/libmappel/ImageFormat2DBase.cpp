/** @file ImageFormat2DBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for ImageFormat2DBase
 */
#include "Mappel/ImageFormat2DBase.h"

namespace mappel {
const ImageFormat2DBase::ImageCoordT ImageFormat2DBase::num_dim=2; /**< Number of image dimensions. */
const ImageFormat2DBase::ImageCoordT ImageFormat2DBase::global_min_size=3; /**< Minimum size along any dimension of the image. */
const ImageFormat2DBase::ImageCoordT ImageFormat2DBase::global_max_size=512; /**< Maximum size along any dimension of the image.  This is insanely big to catch obvious errors */

ImageFormat2DBase::ImageFormat2DBase(const ImageSizeT &size)
    : size(size)
{
    check_size(size);
}

ImageFormat2DBase::ImageCoordT  
ImageFormat2DBase::get_size(IdxT idx) const
{ 
    if(idx>1) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::get_size() idx="<<idx<<" is invalid.";
        throw ParameterValueError(msg.str());
    }
    return size(idx); 
}

void ImageFormat2DBase::set_size(const ImageSizeT &size_)
{
    check_size(size_);
    size = size_;
}

/** @brief Check the size argument for the model
 * 
 */
void ImageFormat2DBase::check_size(const ImageSizeT &size_)
{
    if(arma::any(size_ < global_min_size)) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::check_size: Got Size= "<<size_<<"< Min size="<<global_min_size;
        throw ParameterValueError(msg.str());
    } else if(arma::any(size_ > global_max_size)) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::check_size: Got Size= "<<size_<<"> Max size="<<global_max_size;
        throw ParameterValueError(msg.str());
    } else if(!size_.is_finite()) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::check_size: Got non-finite Size= "<<size_;
        throw ParameterValueError(msg.str());
    }
}

/** @brief Check the shape of a single images is correct for model size
 * 
 */
void ImageFormat2DBase::check_image_shape(const ImageT &im) const
{
    if(im.n_rows != size(1) || im.n_cols != size(0)) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::check_image_shape: Got bad image Size= ["<<im.n_rows<<","<<im.n_cols
           <<"] Expected size=["<<size(1)<<","<<size(0)<<"]";
        throw ArrayShapeError(msg.str());
    }
}

/** @brief Check the shape of a stack of images is correct for model size
 * 
 */
void ImageFormat2DBase::check_image_shape(const ImageStackT &ims) const
{
    if(ims.n_rows != size(1) || ims.n_cols != size(0)) {
        std::ostringstream msg;
        msg<<"ImageFormat2DBase::check_image_shape: Got bad image Size=["<<ims.n_rows<<","<<ims.n_cols<<","<<ims.n_slices
           <<"] Expected size=["<<size(1)<<","<<size(0)<<",...]";
        throw ArrayShapeError(msg.str());
    }
}

StatsT ImageFormat2DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = num_dim;
    stats["imageSize.X"] = size(0);
    stats["imageSize.Y"] = size(1);
    stats["imageNumPixels"] = get_num_pixels();
    stats["imageLimits.Xmin"] = 0;
    stats["imageLimits.Xmax"] = size(0);
    stats["imageLimits.Ymin"] = 0;
    stats["imageLimits.Ymax"] = size(1);
    return stats;
}

} /* namespace mappel */
