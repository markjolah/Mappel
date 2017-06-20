
/** @file display.h
 * @author Mark J. Olah (mjo@cs.unm.edu)
 * @date 03-23-2014
 * @brief 
 *
 */
#ifndef _DISPLAY_H
#define _DISPLAY_H

#include <iostream>
#include <iomanip>
#include <armadillo>

namespace mappel {

extern const char * TERM_BLACK;
extern const char * TERM_RED;
extern const char * TERM_GREEN;
extern const char * TERM_YELLOW;
extern const char * TERM_BLUE;
extern const char * TERM_MAGENTA;
extern const char * TERM_CYAN;
extern const char * TERM_WHITE;
extern const char * TERM_DIM_BLACK;
extern const char * TERM_DIM_RED;
extern const char * TERM_DIM_GREEN;
extern const char * TERM_DIM_YELLOW;
extern const char * TERM_DIM_BLUE;
extern const char * TERM_DIM_MAGENTA;
extern const char * TERM_DIM_CYAN;
extern const char * TERM_DIM_WHITE;


// std::ostream& print_labeled_image(std::ostream &out, const arma::mat &im, const char *color=TERM_WHITE);

template<class ImageT>
std::ostream& print_image(std::ostream &out, const ImageT &im);


template <class Vec>
std::ostream& print_vec_row(std::ostream &out, const Vec &vec, const char *header, int header_width, const char *color=nullptr)
{
    if(color) out<<"\033["<<color<<"m";
    out<<std::setw(header_width)<<std::right<<header;
    out.precision(4);
    out.setf(std::ios::fixed, std::ios::floatfield);
    vec.t().raw_print(out);
    out.unsetf(std::ios::fixed & std::ios::floatfield);
    out.precision(8);
    if(color) out<<"\033[0m";
    return out;
}

} /* namespace mappel */

#endif /* _DISPLAY_H */
