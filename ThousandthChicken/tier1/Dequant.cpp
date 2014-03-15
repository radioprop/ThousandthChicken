/*
 * Dequant.cpp
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#include "Dequant.h"


#ifdef CUDA

/**
 * @brief Subband quantization.
 *
 * @param idata Input tile_comp_data.
 * @param size Width and height of subbnad.
 * @param step_size Step size(deltab).
 */
__global__
void subband_dequantization_lossy(int *idata, int2 isize, type_data *odata, int2 osize, int2 cblk_size, const float convert_factor)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * cblk_size.x;
	int m = j + blockIdx.y * cblk_size.y;
	int in = n + m * isize.x;
	int out = n + m * osize.x;

	while (j < cblk_size.y && m < isize.y)
	{
		while (i < cblk_size.x && n < isize.x)
		{
			odata[out] = ((type_data) ((idata[in] >= 0) ? idata[in] : -(idata[in] & 0x7FFFFFFF))) * convert_factor;
			i += BLOCKSIZEX;
			n = i + blockIdx.x * cblk_size.x;
			in = n + m * isize.x;
			out = n + m * osize.x;
		}
		i = threadIdx.x;
		j += BLOCKSIZEY;
		n = i + blockIdx.x * cblk_size.x;
		m = j + blockIdx.y * cblk_size.y;
		in = n + m * isize.x;
		out = n + m * osize.x;
	}
}

/**
 * @brief Subband dequantization.
 *
 * @param idata Input tile_comp_data.
 * @param size Width and height of subbnad.
 * @param step_size Step size(deltab).
 */
__global__
void subband_dequantization_lossless(int *idata, int2 isize, type_data *odata, int2 osize, int2 cblk_size, const int shift_bits)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = i + blockIdx.x * cblk_size.x;
	int m = j + blockIdx.y * cblk_size.y;
	int in = n + m * isize.x;
	int out = n + m * osize.x;

	while (j < cblk_size.y && m < isize.y)
	{
		while (i < cblk_size.x && n < isize.x)
		{
			odata[out] = (type_data) ((idata[in] >= 0) ? (idata[in] >> shift_bits) : -((idata[in] & 0x7FFFFFFF) >> shift_bits));
			i += BLOCKSIZEX;
			n = i + blockIdx.x * cblk_size.x;
			in = n + m * isize.x;
			out = n + m * osize.x;
		}
		i = threadIdx.x;
		j += BLOCKSIZEY;
		n = i + blockIdx.x * cblk_size.x;
		m = j + blockIdx.y * cblk_size.y;
		in = n + m * isize.x;
		out = n + m * osize.x;
	}
}
#endif


type_subband* Dequant::dequantization(type_subband *sb)
{
#ifdef CUDA
	dim3 blocks;
	dim3 threads;
	cudaError_t error;
#endif
	type_res_lvl *res_lvl = sb->parent_res_lvl;
	type_tile_comp *tile_comp = res_lvl->parent_tile_comp;
	type_image *img = tile_comp->parent_tile->parent_img;
	float convert_factor;
	int shift_bits;
	int max_res_lvl;
	int i;

	/* Lossy */
	if (img->wavelet_type)
	{
		/* Max resolution level */
		max_res_lvl = tile_comp->num_dlvls;
		/* Relative de-quantization step size. Step size is signaled relative to the wavelet coefficient bit depth. */
		convert_factor = sb->step_size
				* ((type_data)(1 << (img->num_range_bits + get_exp_subband_gain(sb->orient) + max_res_lvl - res_lvl->dec_lvl_no)));
		shift_bits = 31 - sb->mag_bits;

		convert_factor = convert_factor / ((type_data)(1 << shift_bits));

		sb->convert_factor = convert_factor;

//		println_var(INFO, "Lossy mag_bits:%d convert_factor:%0.16f shift_bits:%d step_size:%f subband_gain:%d", sb->mag_bits, /*sb->step_size
//				* ((type_data)(1 << (img->num_range_bits + get_exp_subband_gain(sb->orient) + max_res_lvl - res_lvl->dec_lvl_no)))*/sb->convert_factor, shift_bits, sb->step_size, get_exp_subband_gain(sb->orient));

	} else /* Lossless */
	{
		shift_bits = 31 - sb->mag_bits;
		sb->convert_factor = 1 << shift_bits;
//		printf("%d\n", shift_bits);
	}

#ifdef CUDA
	cuda_d_allocate_mem((void **) &(sb->cblks_data_d), sb->width * sb->height/*sb->num_cblks * tile_comp->cblk_w * tile_comp->cblk_h*/ * sizeof(int32_t));

//		printf("%d %d %d\n", sb->num_cblks, tile_comp->cblk_w, tile_comp->cblk_h);
	int32_t *dst;

	for (i = 0; i < sb->num_cblks; i++)
	{
		type_codeblock *cblk = &(sb->cblks[i]);
//				printf("%d %d %d %d %d %d %d\n", cblk->tlx, cblk->tly, cblk->width, cblk->height, sb->width, sb->height, cblk->tlx + cblk->tly * sb->width);
		dst = sb->cblks_data_d + cblk->tlx + cblk->tly * sb->width;
		cuda_memcpy2d_dtd(cblk->data_d, /*cblk->width*/tile_comp->cblk_w * sizeof(int32_t), dst, sb->width * sizeof(int32_t), cblk->width * sizeof(int32_t),
				cblk->height);

		cuda_d_free(cblk->data_d);
	}
#endif
	/* Input and output data */
	int *idata = sb->cblks_data_d;
	type_data *odata = tile_comp->img_data_d + sb->tlx + sb->tly * tile_comp->width;

#ifdef CUDA
	int2 isize;
	int2 osize;
	int2 cblk_size;
	isize = make_int2(sb->width, sb->height);
	osize = make_int2(tile_comp->width, tile_comp->height);
	cblk_size = make_int2(tile_comp->cblk_w, tile_comp->cblk_h);

	/* Number of blocks for parallel reduction */
	blocks = dim3(sb->num_xcblks, sb->num_ycblks);
	/* Number of threads for parallel reduction */
	threads = dim3(BLOCKSIZEX, BLOCKSIZEY);
	if (img->wavelet_type)
	{
		subband_dequantization_lossy<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, sb->convert_factor);
	} else
	{
		subband_dequantization_lossless<<<blocks, threads>>>(idata, isize, odata, osize, cblk_size, shift_bits);
	}
	cudaThreadSynchronize();


	// error report
	if (error = cudaGetLastError())
	printf("Error %s\n", cudaGetErrorString(error));
	cuda_d_free(sb->cblks_data_d);
#endif
	return sb;
}

/**
 * @brief Do dequantization for every subbands from tile.
 * @param tile
 */
void Dequant::dequantize_tile(type_tile *tile)
{
	//	println_start(INFO);

	//	start_measure();

	type_image *img = tile->parent_img;
	type_tile_comp *tile_comp;
	type_res_lvl *res_lvl;
	type_subband *sb;
	int i, j, k;

	for (i = 0; i < img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);
		for (j = 0; j < tile_comp->num_rlvls; j++)
		{
			res_lvl = &(tile_comp->res_lvls[j]);
			for (k = 0; k < res_lvl->num_subbands; k++)
			{
				sb = &(res_lvl->subbands[k]);
				dequantization(sb);
			}
		}
	}

//	save_img(img, "dwt.bmp");

	//	stop_measure(INFO);

	//	println_end(INFO);
}

/**
 * @brief Gets the base 2 exponent of the subband gain.
 * @param orient
 * @return
 */
int Dequant::get_exp_subband_gain(int orient)
{
	return (orient & 1) + ((orient >> 1) & 1);
}


