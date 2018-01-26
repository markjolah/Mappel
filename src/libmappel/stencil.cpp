
/** @file stencil.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-22-2014
 * @brief The stencils for pixel based computations
 */
#include <sstream>
#include "util.h"
#include "stencil.h"
#include "display.h"

namespace mappel {

static const double MIN_INTENSITY = 10.;
static const double sqrt2 = sqrt(2);

double normal_quantile_twosided(double confidence)
{
    if( confidence <=0 || confidence >=1) {
        std::ostringstream msg;
        msg<<"Got bad confidence:"<<confidence<<" should be in (0,1).";
        throw BadValueError(msg.str());
    }
    double p = 1 - (1-confidence)/2.;
    return sqrt2*erf(2*p-1);
}

double normal_quantile_onesided(double confidence)
{
    if( confidence <=0 || confidence >=1) {
        std::ostringstream msg;
        msg<<"Got bad confidence:"<<confidence<<" should be in (0,1).";
        throw BadValueError(msg.str());
    }
    return sqrt2*erf(2*confidence-1);
}


void fill_gaussian_stencil(int size, double stencil[],  double sigma)
{
    double norm=gauss_norm(sigma);
    int stencil_size=2*size-1;
    double val=erf(norm*(0.5-size));
    for(int k=0;k<stencil_size;k++){
        double old_val=val;
        val=erf(norm*(k+1.5-size));
        stencil[k]=0.5*(val-old_val);
    }
}
/*
MatT unit_model_image(const IVecT &size, int pos[], const VecT &sigma)
{
    VecT dx=make_d_stencil(size(0), pos[0]+.5);
    VecT dy=make_d_stencil(size(1), pos[1]+.5);
    VecT X=make_X_stencil(size(0), dx, sigma(0));
    VecT Y=make_X_stencil(size(1), dy, sigma(1));
    return Y*X.t();
}

MatT unit_model_image(const IVecT &size, double pos_x, double pos_y, double sigma_x, double sigma_y)
{
    VecT dx=make_d_stencil(size(0), pos_x);
    VecT dy=make_d_stencil(size(1), pos_y);
    VecT X=make_X_stencil(size(0), dx, sigma_x);
    VecT Y=make_X_stencil(size(1), dy, sigma_y);
    return Y*X.t();
}


CubeT unit_model_HS_image(const IVecT &size, int pos[], double sigmaX, double sigmaY, double sigmaL)
{
    arma::cube im(size(2),size(1),size(0));
    VecT dx=make_d_stencil(size(0), pos[0]+.5);
    VecT dy=make_d_stencil(size(1), pos[1]+.5);
    VecT dz=make_d_stencil(size(2), pos[2]+.5);
    VecT X=make_X_stencil(size(0), dx, sigmaX);
    VecT Y=make_X_stencil(size(1), dy, sigmaY);
    VecT Z=make_X_stencil(size(2), dz, sigmaL);
    for(int k=0; k<size(2); k++) for(int j=0; j<size(1); j++) for(int i=0; i<size(0); i++) { //Col major ordering for armadillo
        im(k,j,i)=X(i)*Y(j)*Z(k);
    }
    return im;
}
*/
/*
VecT estimate_duty_ratios(const MatT &im,const MatT &unit_model_im)
{
    int N=im.n_cols;
    VecT col_sum=arma::sum(im,0).t();
    VecT model_col_sum=arma::sum(unit_model_im,0).t();
    double sum=arma::accu(col_sum);
    double model_sum=arma::accu(model_col_sum);
    VecT duty(N);
    for(int i=0;i<N;i++) {
        double ratio=(col_sum(i)*model_sum)/(sum*model_col_sum(i));
        assert(std::isfinite(ratio));
        duty(i)=restrict_value_range(ratio, 0., 1.);
    }
    return duty;
}

VecT estimate_HS_duty_ratios(const CubeT &im, CubeT &unit_model_im)
{
    using arma::accu;
    int size_x=im.n_cols;
    VecT duty(size_x);
    double sum=accu(im);
    double model_sum=accu(unit_model_im);
    for(int x=0;x<size_x;x++){
        double row_sum=accu(im.tube(arma::span(x,x),arma::span()));
        auto model_xslice=unit_model_im.tube(arma::span(x,x),arma::span());
        double model_row_sum=accu(model_xslice);
        if(sum>=1 && model_row_sum>=0.01*model_sum) {
            double ratio=(row_sum*model_sum)/(sum*model_row_sum);
            assert(std::isfinite(ratio));
            duty(x)=restrict_value_range(ratio, 0., 1.);
        } else {
            duty(x)=0.5;
        }
        model_xslice*=duty(x);
    }
    return duty;
}*/


double gaussian_convolution(int x, int y, const MatT &data, const VecT &Xstencil,const VecT &Ystencil)
{
    int size_x=data.n_cols;
    int size_y=data.n_rows;
    double gauss_val=0., sum=0.;
    for(int i=0; i<size_x; i++) for(int j=0; j<size_y; j++)  {
        double Gxy=Xstencil(i-x+size_x-1)*Ystencil(j-y+size_y-1);
        gauss_val+=Gxy*data(j,i);
        sum+=Gxy;
    }
    gauss_val/=sqrt(sum);  //this approximates the loss of information at the edges
    return gauss_val;
}

void estimate_gaussian_2Dmax(const MatT &data, const VecT &Xstencil,const VecT &Ystencil, int max_pos[], double &min_val)
{
    int size_x=data.n_cols;
    int size_y=data.n_rows;
    double max_gauss=-INFINITY, min_gauss=+INFINITY;
    for(int x=0; x<size_x; x+=2) for(int y=0; y<size_y; y+=2) {    //Gaussian cetered at (x,y)
        double gauss_val=gaussian_convolution(x,y,data,Xstencil,Ystencil);
        if(max_gauss<gauss_val){
            max_pos[0]=x; max_pos[1]=y;
            max_gauss=gauss_val;
        }
        if(min_gauss>gauss_val) min_gauss=gauss_val;
    }
    min_val=  min_gauss==+INFINITY ? 0 : min_gauss; 
}

void refine_gaussian_2Dmax(const MatT &data, const VecT &Xstencil,const VecT &Ystencil, int max_pos[])
{
    int delta=1;
    int size_x=data.n_cols;
    int size_y=data.n_rows;
    MatT gauss_vals(size_y,size_x,arma::fill::zeros);

    double max_gauss=gaussian_convolution(max_pos[0],max_pos[1],data,Xstencil,Ystencil);
    gauss_vals(max_pos[1],max_pos[0])=max_gauss;
    bool new_max_found=true;
    while(new_max_found) {
        new_max_found=false;
        for(int x=max_pos[0]-delta; x<=max_pos[0]+delta; x++) for(int y=max_pos[1]-delta; y<=max_pos[1]+delta; y++)  {
            if (x<0 || y<0 || x>=size_x || y>=size_y) continue;  //Past the edge
            if (gauss_vals(y,x)>0) continue; //Already computed this point
            double gauss_val=gaussian_convolution(x,y,data,Xstencil,Ystencil);
            gauss_vals(y,x)=gauss_val;
            if(gauss_val>max_gauss){
                new_max_found=true;
                max_pos[0]=x;
                max_pos[1]=y;
                max_gauss=gauss_val;
            }
        }
    }
}

double gaussian_3D_convolution(int x, int y, int z, const CubeT &data, const VecFieldT &stencils)
{
    int size_z=data.n_rows;
    int size_y=data.n_cols;
    int size_x=data.n_slices;
    const VecT &stencil_x=stencils(0);
    const VecT &stencil_y=stencils(1);
    const VecT &stencil_z=stencils(2);
    double gauss_val=0., sum=0.;
    for(int k=0; k<size_z; k++) for(int j=0; j<size_y; j++) for(int i=0; i<size_x; i++) { //column mjor ordering of data
        double Gxyz=stencil_x(i-x+size_x-1) * stencil_y(j-y+size_y-1) * stencil_z(k-z+size_z-1);
        assert(Gxyz>=0);
        gauss_val+=Gxyz*data(k,j,i);
        sum+=Gxyz;
    }
    gauss_val/=sqrt(sum);  //this approximates the loss of information at the edges
    return gauss_val;
}



void estimate_gaussian_3Dmax(const CubeT &data, const VecFieldT &stencils, int max_pos[], double &min_val)
{
    double max_gauss=-INFINITY;
    double min_gauss=+INFINITY;
    int size_z=data.n_rows;
    int size_y=data.n_cols;
    int size_x=data.n_slices;
    //Gaussian cetered at (x,y,z)
    for(int z=0; z<size_z; z+=2) for(int y=0; y<size_y; y+=2) for(int x=0; x<size_x; x+=2) {
        double gauss_val=gaussian_3D_convolution(x,y,z,data,stencils);
        assert(gauss_val>=0);
        if(max_gauss<gauss_val){
            max_pos[0]=x; max_pos[1]=y; max_pos[2]=z;
            max_gauss=gauss_val;
        }
        if(min_gauss>gauss_val) min_gauss=gauss_val;
    }
    min_val=  min_gauss==+INFINITY ? 0 : min_gauss;
}

void refine_gaussian_3Dmax(const CubeT &data, const VecFieldT &stencils, int max_pos[])
{
    int delta=1;
    int size_z=data.n_rows;
    int size_y=data.n_cols;
    int size_x=data.n_slices;
    CubeT gauss_vals(size_x,size_y,size_z,arma::fill::zeros);

    assert(max_pos[0]>=0 && max_pos[1]>=0 && max_pos[2]>=0);
    assert(max_pos[0]<size_x && max_pos[1]<size_y && max_pos[2]<size_z);
    double max_gauss=gaussian_3D_convolution(max_pos[0],max_pos[1],max_pos[2],data,stencils);
    gauss_vals(max_pos[0],max_pos[1],max_pos[2])=max_gauss;
    bool new_max_found=true;
    while(new_max_found) {
        new_max_found=false;
        for(int z=max_pos[2]-delta; z<=max_pos[2]+delta; z++) {
            if (z<0 || z>=size_z) continue;  //Past the edge
            for(int y=max_pos[1]-delta; y<=max_pos[1]+delta; y++) {
                if (y<0 || y>=size_y) continue;  //Past the edge
                for(int x=max_pos[0]-delta; x<=max_pos[0]+delta; x++) {
                    if (x<0 || x>=size_x) continue;  //Past the edge
                    if (gauss_vals(x,y,z)>0) continue; //Already computed this point
                    double gauss_val=gaussian_3D_convolution(x,y,z,data,stencils);
                    gauss_vals(x,y,z)=gauss_val;
                    if(gauss_val>max_gauss){
                        new_max_found=true;
                        max_pos[0]=x;
                        max_pos[1]=y;
                        max_pos[2]=z;
                        max_gauss=gauss_val;
                    }
                }
            }
        }
    }
}


double estimate_background(const MatT &im, const MatT &unit_model_im, double min_bg)
{
    arma::umat model_mask = unit_model_im<0.0001;
    int mask_size=arma::accu(model_mask);
    int min_mask_size=std::max(8.,im.n_elem*0.1);
    double bg= arma::accu(im % model_mask)/mask_size;
    if(mask_size<min_mask_size){
        return min_bg;
    }
    return std::max(0.,bg);
}



double estimate_intensity(const MatT &im, const MatT &unit_model_im, double bg)
{
    return std::max(MIN_INTENSITY, arma::accu(im-bg)/arma::accu(unit_model_im));
}

double estimate_background(const CubeT &im, const CubeT &unit_model_im)
{
    const double threshold=1E-5;
    arma::ucube model_mask = unit_model_im<threshold;
    int mask_size=arma::accu(model_mask);
    double bg=arma::accu(im % model_mask)/mask_size;
    return std::max(0.,bg);
}


double estimate_intensity(const CubeT &im, const CubeT &unit_model_im, double bg)
{
    return std::max(MIN_INTENSITY,arma::accu(im-bg)/arma::accu(unit_model_im));
}

} /* namespace mappel */
