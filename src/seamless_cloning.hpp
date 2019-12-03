#ifndef SEAMLESSCLONE_H_H
#define SEAMLESSCLONE_H_H 
 
 
//#include "precomp.hpp"
//#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdlib.h>
#include <complex>
#include "math.h"
 
using namespace std;
using namespace cv;
 
namespace customCV {
 
	class Cloning
	{
 
	public:
 
		//output: 每个通道的合成结果数组
		//rbgx_channel, rgby_channel是gxx， gyy 分通道结果
		vector <Mat> rgb_channel, rgbx_channel, rgby_channel, output;
 
		//smask是source图片的mask， smask1是smask取反的结果
		//grx, gry 是dst图片的梯度。 grx32， gry32是smask1区域的梯度
		//sgx, sgy 是source图片的梯度。 srx32, sry32是smask区域的梯度
		Mat grx, gry, sgx, sgy, srx32, sry32, grx32, gry32, smask, smask1;
		void init_var(Mat &I, Mat &wmask);
		void initialization(Mat &I, Mat &mask, Mat &wmask);
		void scalar_product(Mat mat, float r, float g, float b);
		void array_product(Mat mat1, Mat mat2, Mat mat3);
		void poisson(Mat &I, Mat &gx, Mat &gy, Mat &sx, Mat &sy);
		void evaluate(Mat &I, Mat &wmask, Mat &cloned);
		void getGradientx(const Mat &img, Mat &gx);
		void getGradienty(const Mat &img, Mat &gy);
		void lapx(const Mat &img, Mat &gxx);
		void lapy(const Mat &img, Mat &gyy);
		void dst(double *mod_diff, double *sineTransform, int h, int w);
		void idst(double *mod_diff, double *sineTransform, int h, int w);
		void transpose(double *mat, double *mat_t, int h, int w);
		void solve(const Mat &img, double *mod_diff, Mat &result);
		void poisson_solver(const Mat &img, Mat &gxx, Mat &gyy, Mat &result);
		void normal_clone(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, int num);
		void local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul, float green_mul, float blue_mul);
		void illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta);
		void texture_flatten(Mat &I, Mat &mask, Mat &wmask, double low_threshold, double high_threhold, int kernel_size, Mat &cloned);
	};
 
 
 
 
	void seamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags);
	void colorChange(InputArray _src, InputArray _mask, OutputArray _dst, float r, float g, float b);
	void illuminationChange(InputArray _src, InputArray _mask, OutputArray _dst, float a, float b);
	void textureFlattening(InputArray _src, InputArray _mask, OutputArray _dst, double low_threshold, double high_threshold, int kernel_size);

}

#endif