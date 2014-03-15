/**
 * @file preprocess_gpu.cu
 *
 * @brief Performes image preprocessing.
 *
 * It is the first step of the encoder workflow and the last one of the decoder workflow.
 * Changes image's color space to YUV.
 * Includes modes for lossy and lossless colorspace transformation.
 *
 */

#include <stdio.h>

#include "../types/image_types.h"
#include "../print_info/print_info.h"


#include "preprocessing_constants.h"

#define __global__
#define __device__
#define __shared__

/**
 * @brief CUDA kernel for DC level shifting coder.
 *
 * It performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img The image data.
 * @param size Number of pixels in each component (width x height).
 * @param level_shift Level shift.
 */
void __global__ fdc_level_shift_kernel(type_data *idata, const uint16_t width, const uint16_t height, const int level_shift) {
#ifdef CUDA
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			idata[idx] = idata[idx] - (1 << level_shift);
			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
#endif
	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		img[idx] -= 1 << level_shift;
	}*/
}

int __device__ clamp_val(int val, int min, int max)
{
	if(val < min)
		return min;
	if(val > max)
		return max;
	return val;
}

/**
 * @brief CUDA kernel for DC level shifting decoder.
 *
 * It performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just adds 128. (so the data range is [-128 ; 128] ).
 *
 * @param img The image data.
 * @param size Number of pixels in each component (width x height).
 * @param level_shift Level shift.
 */
void __global__ idc_level_shift_kernel(type_data *idata, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
#ifdef CUDA
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	int cache;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			cache = idata[idx] + (1 << level_shift);
			idata[idx] = clamp_val(cache, min, max);
//			idata[idx] = idata[idx] + (1 << level_shift);
			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
#endif
	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		img[idx] -= 1 << level_shift;
	}*/
}

/**
 * @brief CUDA kernel for the Reversible Color Transformation (lossless) coder.
 *
 * Before colorspace transformation it performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img_r 1D array with RED component of the image.
 * @param img_g 1D array with GREEN component of the image.
 * @param img_b 1D array with BLUE component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ rct_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift) {
#ifdef CUDA
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	int r, g, b;
	int y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			b = img_b[idx] - (1 << level_shift);
			g = img_g[idx] - (1 << level_shift);
			r = img_r[idx] - (1 << level_shift);

			y = (r + 2*g + b)>>2;
			u = b - g;
			v = r - g;

			img_r[idx] = (type_data)y;
			img_b[idx] = (type_data)u;
			img_g[idx] = (type_data)v;

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
#endif
	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		int r, g, b;
		int y, u, v;

		b = img_b[idx] - 128;
		g = img_g[idx] - 128;
		r = img_r[idx] - 128;

		y = (r + 2*g + b)>>2;
		u = b - g;
		v = r - g;

		img_b[idx] = (type_data)y;
		img_g[idx] = (type_data)u;
		img_r[idx] = (type_data)v;
	}*/
}

/**
 * @brief CUDA kernel for the Reversible Color Transformation (lossless) decoder.
 *
 *
 * After colorspace transformation it performs dc level shifting to shift data back to it's unsigned form,
 * just adds 128. (so the data range is [0 ; 256] ).
 *
 * @param img_r 1D array with V component of the image.
 * @param img_g 1D array with U component of the image.
 * @param img_b 1D array with Y component of the image.
 * @param size Number of pixels in each component (width x height).
 */
//void __global__ tcr_kernel(type_data *img_r, type_data *img_g, type_data *img_b, long int size) {
void __global__ tcr_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
#ifdef CUDA
	    int i = threadIdx.x;
		int j = threadIdx.y;
		int n = i + blockIdx.x * TILE_SIZEX;
		int m = j + blockIdx.y * TILE_SIZEY;
		int idx = n + m * width;
		int r, g, b;
		int y, u, v;

		while(j < TILE_SIZEY && m < height)
		{
			while(i < TILE_SIZEX && n < width)
			{
				y = img_r[idx];
				u = img_g[idx];
				v = img_b[idx];


				g = y - ((v + u)>>2);
				r = (v + g);
				b = (u + g);

				b = (type_data)b + (1 << level_shift);
				g = (type_data)g + (1 << level_shift);
				r = (type_data)r + (1 << level_shift);

				img_r[idx] = clamp_val(r, min, max);
				img_b[idx] = clamp_val(g, min, max);
				img_g[idx] = clamp_val(b, min, max);

//				img_r[idx] = (type_data)b + (1 << level_shift);
//				img_b[idx] = (type_data)g + (1 << level_shift);
//				img_g[idx] = (type_data)r + (1 << level_shift);

				i += BLOCK_SIZE;
				n = i + blockIdx.x * TILE_SIZEX;
				idx = n + m * width;
			}

			i = threadIdx.x;
			j += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			m = j + blockIdx.y * TILE_SIZEY;
			idx = n + m * width;
		}
#endif
/*
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		y = img_b[idx];
		u = img_g[idx];
		v = img_r[idx];

		g = y - floor((u + v) / 4);
		r = (v + g);
		b = (u + g);

		img_b[idx] = b + 128;
		img_g[idx] = g + 128;
		img_r[idx] = r + 128;
	}
*/
}

/**
 * @brief CUDA kernel for the Irreversible Color Transformation (lossy) coder.
 *
 * Before colorspace transformation it performs dc level shifting to centralize data around 0. Doesn't use
 * any sophisticated algorithms for this, just subtracts 128. (so the data range is [-128 ; 128] ).
 *
 * @param img_r 1D array with RED component of the image.
 * @param img_g 1D array with GREEN component of the image.
 * @param img_b 1D array with BLUE component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ ict_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift) {
#ifdef CUDA
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	type_data r, g, b;
	type_data y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			b = img_r[idx] - (1 << level_shift);
			g = img_g[idx] - (1 << level_shift);
			r = img_b[idx] - (1 << level_shift);

			y = Wr*r + Wg*g + Wb*b;
			u = -0.16875f * r - 0.33126f * g + 0.5f * b;
//			u = (Umax * ((b - y) / (1 - Wb)));
			v = 0.5f * r - 0.41869f * g - 0.08131f * b;
//			v = (Vmax * ((r - y) / (1 - Wr)));

			img_r[idx] = y;
			img_g[idx] = u;
			img_b[idx] = v;

/*			img_r[idx] = y;
			img_g[idx] = u;
			img_b[idx] = v;*/

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
#endif
	/*int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		b = img_b[idx] - 128;
		g = img_g[idx] - 128;
		r = img_r[idx] - 128;

		y = Wr*r + Wg*g + Wb*b;
		u = (Umax * ((b - y) / (1 - Wb)));
		v = (Vmax * ((r - y) / (1 - Wr)));

		img_b[idx] = y;
		img_g[idx] = u;
		img_r[idx] = v;
	}*/
}

/**
 * @brief CUDA kernel for the Irreversible Color Transformation (lossy) decoder.
 *
 *
 * After colorspace transformation it performs dc level shifting to shift data back to it's unsigned form,
 * just adds 128. (so the data range is [0 ; 256] ).
 *
 * @param img_r 1D array with V component of the image.
 * @param img_g 1D array with U component of the image.
 * @param img_b 1D array with Y component of the image.
 * @param size Number of pixels in each component (width x height).
 */
void __global__ tci_kernel(type_data *img_r, type_data *img_g, type_data *img_b, const uint16_t width, const uint16_t height, const int level_shift, const int min, const int max) {
#ifdef CUDA
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * TILE_SIZEX;
	int m = j + blockIdx.y * TILE_SIZEY;
	int idx = n + m * width;
	type_data r, g, b;
	type_data y, u, v;

	while(j < TILE_SIZEY && m < height)
	{
		while(i < TILE_SIZEX && n < width)
		{
			y = img_b[idx];
			u = img_g[idx];
			v = img_r[idx];

			type_data r_tmp = v*( (1 - Wr)/Vmax );
			type_data b_tmp = u*( (1 - Wb)/Umax );

			r = y + r_tmp;
			b = y + b_tmp;
			g = y - (Wb/Wg) * r_tmp - (Wr/Wg) * b_tmp;

			b = (type_data)b + (1 << level_shift);
			g = (type_data)g + (1 << level_shift);
			r = (type_data)r + (1 << level_shift);

			img_b[idx] = clamp_val(b, min, max);
			img_g[idx] = clamp_val(g, min, max);
			img_r[idx] = clamp_val(r, min, max);

//			img_b[idx] = b + (1 << level_shift);
//			img_g[idx] = g + (1 << level_shift);
//			img_r[idx] = r + (1 << level_shift);

			i += BLOCK_SIZE;
			n = i + blockIdx.x * TILE_SIZEX;
			idx = n + m * width;
		}
		i = threadIdx.x;
		j += BLOCK_SIZE;
		n = i + blockIdx.x * TILE_SIZEX;
		m = j + blockIdx.y * TILE_SIZEY;
		idx = n + m * width;
	}
#endif

/*
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if(idx < size) {
		type_data r, g, b;
		type_data y, u, v;

		y = img_b[idx];
		u = img_g[idx];
		v = img_r[idx];

		type_data r_tmp = v*( (1 - Wr)/Vmax );
		type_data b_tmp = u*( (1 - Wb)/Umax );

		r = y + r_tmp;
		b = y + b_tmp;
		g = y - (Wb/Wg) * r_tmp - (Wr/Wg) * b_tmp;
	
		img_b[idx] = b + 128;
		img_g[idx] = g + 128;
		img_r[idx] = r + 128;
	}
*/
}
