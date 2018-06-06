
/** @file display.cpp
 * @author Mark J. Olah (mjo@cs.unm DOT edu)
 * @date 03-23-2014
 * @brief 
 *
 */

#include "Mappel/display.h"

namespace mappel {
    
const char * TERM_BLACK="1;30";
const char * TERM_RED="1;31";
const char * TERM_GREEN="1;32";
const char * TERM_YELLOW="1;33";
const char * TERM_BLUE="1;34";
const char * TERM_MAGENTA="1;35";
const char * TERM_CYAN="1;36";
const char * TERM_WHITE="1;37";
const char * TERM_DIM_BLACK="0;30";
const char * TERM_DIM_RED="0;31";
const char * TERM_DIM_GREEN="0;32";
const char * TERM_DIM_YELLOW="0;33";
const char * TERM_DIM_BLUE="0;34";
const char * TERM_DIM_MAGENTA="0;35";
const char * TERM_DIM_CYAN="0;36";
const char * TERM_DIM_WHITE="0;37";

using namespace std;

const char * 
lambda_term_color(int size, int Lidx)
{
    if (size<=7) {
        switch(Lidx) {
            case 0: return TERM_MAGENTA;
            case 1: return TERM_BLUE;
            case 2: return TERM_CYAN;
            case 3: return TERM_GREEN;
            case 4: return TERM_YELLOW;
            case 5: return TERM_RED;
            default: return TERM_DIM_RED;
        }
    } else if (size<=10) {
        switch(Lidx) {
            case 0: return TERM_DIM_BLUE;
            case 1: return TERM_DIM_MAGENTA;
            case 2: return TERM_MAGENTA;
            case 3: return TERM_BLUE;
            case 4: return TERM_CYAN;
            case 5: return TERM_GREEN;
            case 6: return TERM_YELLOW;
            case 7: return TERM_DIM_YELLOW;
            case 8: return TERM_RED;
            default: return TERM_DIM_RED;
        }
    } else {
        int i=((float)Lidx/size)*14.;
        switch(i) {
            case 0: return TERM_WHITE;
            case 1: return TERM_DIM_WHITE;
            case 2: return TERM_MAGENTA;
            case 3: return TERM_DIM_MAGENTA;
            case 4: return TERM_BLUE;
            case 5: return TERM_DIM_BLUE;
            case 6: return TERM_CYAN;
            case 7: return TERM_DIM_CYAN;
            case 8: return TERM_GREEN;
            case 9: return TERM_DIM_GREEN;
            case 10: return TERM_YELLOW;
            case 11: return TERM_DIM_YELLOW;
            case 12: return TERM_RED;
            default: return TERM_DIM_RED;
        }
    }
}

static const char *colstr1="     \\";
static const char *colstr2="+y|||_";
static const char *colstr3="     /";

ostream& print_centered_title(ostream &out,char fill, int width, const char *title=nullptr)
{
    if(title){
        int len=strnlen(title, width);
        int sp=(width+len+1)/2;
        out<<setw(sp)<<setfill(fill)<<right<<title<<setw(width-sp)<<""<<setfill(' ');
    } else {
        out<<setw(width)<<setfill(fill)<<""<<setfill(' ');
    }
    return out;
}

ostream& print_labeled_image(ostream &out, const arma::mat &im, const char *title,  const char *color)
{
    bool int_vals=not arma::any((arma::vectorise(im)<1) % (arma::vectorise(im)>0) );
    int lmargin=10;
    int cellw;
    if(int_vals){
        if (arma::any(arma::vectorise(im)>=1000)) {
            cellw=5;
        } else {
            cellw=4;
        }
    } else {
        cellw=7;
    }
    int imwidth=cellw*im.n_cols+2;
    out<<setw(lmargin+20)<<"+x------->"<<endl;
    out<<setw(lmargin+1)<<"";
    for(unsigned j=0; j<im.n_cols; j++) out<<setw(cellw)<<j;
    out<<endl;
    out<<setw(lmargin)<<"(0,0)  ";
    print_centered_title(out,'=',imwidth,title)<<endl;
    for(unsigned i=0; i<im.n_rows; i++) {
        out<<setw(lmargin-5);
        if(i<strlen(colstr1)) out<<colstr1[i]<<colstr2[i]<<colstr3[i];
        else out<<setw(lmargin-3)<<"";
        out<<setw(2)<<i<<" |";
        for(unsigned j=0; j<im.n_cols; j++) {
            if(color) out<<"\033["<<color<<"m";
            if(int_vals){
                out<<setw(cellw)<<(int)im(i,j);
            } else {
                printf("%*.*f",cellw,cellw-3,im(i,j));
//                 out<<setw(cellw)<<setprecision(cellw-3)<<im(i,j);
            }
            if(color) out<<"\033[0m";
        }
        out<<'|'<<endl;
    }
    out<<setw(lmargin)<<"";
    print_centered_title(out,'=',imwidth)<<" ("<<im.n_rows<<","<<im.n_cols<<")"<<endl;
    return out;
}

template<>
std::ostream& print_image(std::ostream &out, const arma::vec &im)
{
    print_labeled_image(out, im, "IMAGE", TERM_WHITE);
    return out;
}


template<>
std::ostream& print_image(std::ostream &out, const arma::mat &im)
{
    print_labeled_image(out, im, "IMAGE", TERM_WHITE);
    return out;
}

template<>
std::ostream& print_image(std::ostream &out, const arma::cube &im)
{
    char str[64];
    int size=im.n_slices;
    int w=4*im.n_cols+20;
    print_centered_title(out,'#',w,"BEGIN HYPERSPECTRAL IMAGE")<<endl;
    for(int n=0; n<size; n++) {
        snprintf(str,64," Lambda:[%i] ",n);
        print_labeled_image(out, im.slice(n),str, lambda_term_color(size, n));
    }
    print_centered_title(out,'#',w,"END HYPERSPECTRAL IMAGE")<<endl;
    return out;
}

} /* namespace mappel */
