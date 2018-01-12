/** @file ImageFormat1DBase.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for ImageFormat1DBase.
 *
 * The virtual base class for all point 2D image based emitter Models and Objectives
 */

#ifndef _IMAGEFORMATBASE_H
#define _IMAGEFORMATBASE_H

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
    using ImageSizeT = uint32_t;
    using ImageSizeVecT = arma::Col<ImageSizeT>;
    using ImagePixelT = double;
    template<class T> using ImageShapeT = arma::Col<T>;
    template<class T> using ImageStackShapeT = arma::Mat<T>;
    using ImageT = ImageShapeT<ImagePixelT>; /**< A type to represent image data*/
    using ImageStackT = ImageStackShapeT<ImagePixelT>; /**< A type to represent image data stacks */

    constexpr static IdxT num_dim = 1;
    constexpr static IdxT min_size = 3; /**< Minimum size along any dimension of the image.  Prevents "too small to be meaningfull" images. */
    /* Model parameters */
    ImageSizeT size; /**< The number of pixels >0 */

    ImageFormat1DBase(ImageSizeT size_);
    ImageFormat1DBase(const ImageSizeVecT &size_);
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(IdxT n) const;
    IdxT size_image_stack(const ImageStackT &stack) const;
    ImageT get_image_from_stack(const ImageStackT &stack, IdxT n) const;
    
    ImageT model_image(
private:
    static void check_size(ImageSizeT size_);
};


} /* namespace mappel */

#endif /* _IMAGEFORMAT1DBASE_H */
